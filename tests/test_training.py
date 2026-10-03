from dataclasses import replace

import pytest
import torch

from ncpu_computer_1d import (
    ExperimentConfig,
    ModelConfig,
    NeuralCellularAutomaton,
    TapeLayout,
    TestCase,
    Trainer,
    TrainingConfig,
    binary_tasks,
    evaluate_cases,
    infer,
    learning_rate_at,
    load_model,
    sample_trials,
    supervised_loss,
    validate_experiment,
    validation_trials,
)
from ncpu_computer_1d.tasks import MultiTaskDataset


def tiny_config(**model_changes):
    return ExperimentConfig(
        model=ModelConfig(
            program_channels=2,
            computation_channels=2,
            hidden_size=8,
            fire_rate=0.8,
            **model_changes,
        ),
        training=TrainingConfig(
            updates=2,
            batch_size_per_task=3,
            base_tape_slots=(4, 6),
            base_input_max_lengths=(1, 3),
            tape_variation=0.2,
            input_variation=0.2,
            free_steps_per_tape_slot=0.5,
            time_variation=0.2,
            supervision_ratio=1.5,
            validation_every=1,
            checkpoint_every=1,
            perception_noise_start=0.001,
            perception_noise_end=0.001,
            device="cpu",
        ),
        test_cases=(TestCase("id", 6, 3, 3, 4),),
    )


def test_full_tape_weighted_loss_window_and_gradients():
    rollout = torch.randn(4, 6, 3, 5, requires_grad=True)
    target = torch.zeros(4, 5)
    indices = torch.tensor([1, 0, 1, 0])
    losses = supervised_loss(
        rollout, target, TapeLayout(5), 1, 1, 3, indices, 2, (1, 2)
    )
    per_example = rollout[:, 2:5, 1].square().mean((1, 2))
    reference = (
        per_example[indices == 0].mean() + 2 * per_example[indices == 1].mean()
    ) / 3
    torch.testing.assert_close(losses.total, reference)
    a = torch.autograd.grad(losses.total, rollout, retain_graph=True)[0]
    b = torch.autograd.grad(reference, rollout)[0]
    torch.testing.assert_close(a, b)
    assert torch.count_nonzero(a[:, :2]) == torch.count_nonzero(a[:, 5:]) == 0
    assert bool((a[:, 2:5, 1, -1] != 0).all())  # blanks carry supervision
    with pytest.raises(ValueError):
        supervised_loss(rollout, target, TapeLayout(5), 1, 0, 0)


def test_sampling_resamples_inputs_only_and_obeys_both_capacities():
    training = replace(
        tiny_config().training,
        base_tape_slots=(5, 7),
        base_input_max_lengths=(2, 4),
        tape_variation=0.3,
        input_variation=0.3,
    )
    rng = torch.Generator().manual_seed(3)
    tape_values, input_values = set(), set()
    for _ in range(80):
        for trial in sample_trials(training, ("reverse", "append_0"), rng):
            assert trial.input_max_length + 1 <= trial.tape_slots - 1
            assert trial.free_steps >= 1 and trial.supervision_steps >= 1
            tape_values.add(trial.tape_slots)
            input_values.add(trial.input_max_length)
    assert len(tape_values) > 3 and len(input_values) > 2
    trials = validation_trials(training, ("reverse", "append_0"))
    assert [t.tape_slots for t in trials] == [5, 7]
    assert [t.input_max_length for t in trials] == [2, 4]
    invalid = replace(training, base_input_max_lengths=(3, 5), input_variation=0)
    with pytest.raises(ValueError):
        sample_trials(invalid, ("append_0",), rng)


def test_accumulated_gradients_equal_mean_loss_across_shapes(monkeypatch):
    config = tiny_config()
    config = replace(
        config,
        training=replace(
            config.training,
            tape_variation=0,
            input_variation=0,
            time_variation=0,
            grad_clip=None,
            perception_noise_start=0,
            perception_noise_end=0,
        ),
        model=replace(config.model, fire_rate=1),
    )
    trainer = Trainer(config, (("reverse", 2.0), ("bit_not", 1.0)))
    with torch.no_grad():
        trainer.model.rule.output.weight.normal_(std=0.02)
    reference = NeuralCellularAutomaton(config.model, trainer.task_names)
    reference.load_state_dict(trainer.model.state_dict())
    trial_rng = torch.Generator()
    trial_rng.set_state(trainer.trial_generator.get_state())
    data_rng = torch.Generator()
    data_rng.set_state(trainer.data_generator.get_state())
    losses, lengths = [], []
    for trial in sample_trials(config.training, trainer.task_names, trial_rng):
        dataset = MultiTaskDataset.from_tasks(
            binary_tasks(trainer.task_names, trial.input_max_length), trial.tape_slots
        )
        inputs, targets, _, tasks = dataset.balanced_sample(
            config.training.batch_size_per_task, data_rng
        )
        rollout = reference(reference.initial_state(inputs, tasks), trial.steps)
        lengths.append(inputs.shape[-1])
        losses.append(
            supervised_loss(
                rollout,
                targets,
                TapeLayout(trial.tape_slots),
                config.model.io_channel,
                trial.free_steps,
                trial.supervision_steps,
                tasks,
                2,
                trainer.task_weights,
            ).total
        )
    (torch.stack(losses).mean()).backward()
    monkeypatch.setattr(trainer.optimizer, "step", lambda: None)
    metrics = trainer.train_step()
    assert sorted(lengths) == [4, 6]
    assert metrics.loss == pytest.approx(float(torch.stack(losses).mean()), rel=1e-6)
    for a, b in zip(trainer.model.parameters(), reference.parameters()):
        torch.testing.assert_close(a.grad, b.grad)


@pytest.mark.parametrize("radius", [1, 2])
def test_smoke_validation_checkpoint_exact_resume_and_fixed_evaluation(
    tmp_path, radius
):
    config = tiny_config(radius=radius)
    specs = (("reverse", 2.0), ("bit_not", 1.0))
    validate_experiment(config, ("reverse", "bit_not"))
    assert ExperimentConfig.from_dict(config.to_dict()) == config
    trainer = Trainer(config, specs)
    trainer.train_step()
    rng_before = torch.get_rng_state().clone()
    cuda_before = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    trainer.validation_metrics()
    assert trainer.model.training
    assert torch.equal(torch.get_rng_state(), rng_before)
    for before, after in zip(cuda_before, torch.cuda.get_rng_state_all()):
        assert torch.equal(before, after)
    path = tmp_path / "state.pt"
    trainer.save(path)
    first = trainer.train_step()
    expected = {key: value.clone() for key, value in trainer.model.state_dict().items()}
    restored = Trainer.from_checkpoint(path, specs, "cpu")
    second = restored.train_step()
    assert first.to_dict() == second.to_dict()
    for key, value in expected.items():
        torch.testing.assert_close(
            value, restored.model.state_dict()[key], rtol=0, atol=0
        )
    model, loaded, checkpoint = load_model(path, "cpu")
    assert loaded == config
    before = torch.get_rng_state().clone()
    cuda_before = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    results = evaluate_cases(model, model.task_names, config.test_cases, batch_size=3)
    assert torch.equal(torch.get_rng_state(), before)
    for old, current in zip(cuda_before, torch.cuda.get_rng_state_all()):
        assert torch.equal(old, current)
    assert results[0].examples == 8
    assert results[0].step_start == 4 and results[0].step_end == 7
    assert results[-1].task == "aggregate"
    result = infer(model, "101", tape_slots=6, task_name="reverse", steps=2)
    assert result.values.shape == (6,)
    with pytest.raises(ValueError):
        Trainer.from_checkpoint(path, (("reverse", 1.0), ("bit_not", 1.0)))
    checkpoint["format_version"] = 8
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="incompatible"):
        load_model(path, "cpu")


def test_final_tests_error_instead_of_truncating_append_output():
    model = NeuralCellularAutomaton(ModelConfig(), ("append_0",))
    with pytest.raises(ValueError, match="output"):
        evaluate_cases(model, ("append_0",), (TestCase("bad", 4, 3, 1, 1),))


def test_flags_lr_and_fit_checkpoint_history(tmp_path):
    config = tiny_config()
    assert learning_rate_at(config.training, 0) == config.training.lr_points[0][1]
    assert learning_rate_at(config.training, 1) == pytest.approx(
        config.training.lr_points[-1][1]
    )
    trainer = Trainer(config, (("reverse", 2.0),))
    history = trainer.fit(tmp_path, progress_every=1)
    assert len(history) == 2
    assert all("reverse" in entry["validation_accuracies"] for entry in history)
    assert (tmp_path / "best.pt").is_file() and (tmp_path / "latest.pt").is_file()
    for train_rule, train_program in [(True, False), (False, True)]:
        configured = replace(
            config,
            training=replace(
                config.training, train_rule=train_rule, train_program=train_program
            ),
        )
        trainer = Trainer(configured, (("reverse", 1.0),))
        assert all(
            p.requires_grad == train_rule for p in trainer.model.shared_parameters
        )
        assert all(
            p.requires_grad == train_program for p in trainer.model.program_parameters
        )
        trainer.train_step()


@pytest.mark.parametrize("placement", ["periodic", "prefix"])
def test_contained_program_sampling_validation_and_checkpoint(tmp_path, placement):
    config = tiny_config(program_length=5, program_placement=placement)
    validate_experiment(config, ("reverse", "copy"))
    rng = torch.Generator().manual_seed(8)
    values = []
    for _ in range(40):
        trials = sample_trials(
            config.training, ("reverse", "copy"), rng, minimum_tape_slots=5
        )
        values.extend(trial.tape_slots for trial in trials)
        assert all(trial.tape_slots >= 5 for trial in trials)
    assert set(values) == {5, 6, 7}
    validation = validation_trials(
        config.training, ("reverse", "copy"), minimum_tape_slots=5
    )
    assert [trial.tape_slots for trial in validation] == [5, 6]
    assert validation[0].free_steps == 3
    specs = (("reverse", 2.0), ("copy", 1.0))
    trainer = Trainer(config, specs)
    trainer.train_step()
    path = tmp_path / "program.pt"
    trainer.save(path)
    first = trainer.train_step()
    expected = {
        name: value.clone() for name, value in trainer.model.state_dict().items()
    }
    restored = Trainer.from_checkpoint(path)
    assert restored.model.programs.shape == (2, 5, 2)
    second = restored.train_step()
    assert first.to_dict() == second.to_dict()
    for name, value in expected.items():
        torch.testing.assert_close(
            value, restored.model.state_dict()[name], rtol=0, atol=0
        )
    invalid = replace(config, model=replace(config.model, program_length=6))
    with pytest.raises(ValueError, match="tape range"):
        invalid.validate()
    model = NeuralCellularAutomaton(config.model, ("reverse", "copy"))
    with pytest.raises(ValueError, match="cannot contain"):
        evaluate_cases(model, model.task_names, (TestCase("fixed", 4, 1, 1, 2),))
    with pytest.raises(ValueError, match="cannot contain"):
        replace(config, test_cases=(TestCase("fixed", 4, 1, 1, 2),)).validate()


def test_compact_loss_and_unbalanced_task_weights_match_full_reference():
    rollout = torch.randn(5, 7, 3, 4, dtype=torch.float64, requires_grad=True)
    targets = torch.randn(5, 4, dtype=torch.float64)
    indices = torch.tensor([0, 1, 0, 1, 0])
    full = supervised_loss(rollout, targets, TapeLayout(4), 1, 2, 4, indices, 2, (1, 2))
    compact = supervised_loss(
        rollout[:, 3:, 1],
        targets,
        TapeLayout(4),
        1,
        2,
        4,
        indices,
        2,
        (1, 2),
        start_step=3,
    )
    errors = (rollout[:, 3:, 1] - targets[:, None]).square().mean((1, 2))
    reference = (errors[indices == 0].mean() + 2 * errors[indices == 1].mean()) / 3
    torch.testing.assert_close(compact.total, reference)
    torch.testing.assert_close(full.total, compact.total)
    a = torch.autograd.grad(compact.total, rollout, retain_graph=True)[0]
    b = torch.autograd.grad(reference, rollout)[0]
    torch.testing.assert_close(a, b)
    for bad_indices in (torch.tensor([0, 0, 0, 0, 0]), torch.tensor([0, 1, 0, 1, 2])):
        with pytest.raises(ValueError):
            supervised_loss(rollout, targets, TapeLayout(4), 1, 2, 4, bad_indices, 2)


def test_datasets_are_lightweight_and_independent_of_time():
    from ncpu_computer_1d import TrialSpec

    trainer = Trainer(tiny_config(), (("reverse", 1.0),))
    a = TrialSpec(4, 1, 2, 3)
    b = TrialSpec(4, 1, 3, 4)
    assert trainer._datasets(a) == trainer._datasets(b)
    assert trainer._datasets(a) is not trainer._datasets(a)


def test_larger_curriculum_bounds_and_fixed_extrapolation():
    from ncpu_computer_1d import round_half_up, varied_bounds

    config = ExperimentConfig()
    config.validate()
    training = config.training
    input_ceiling = max(
        varied_bounds(base, training.input_variation)[1]
        for base in training.base_input_max_lengths
    )
    tape_ceiling = max(
        varied_bounds(base, training.tape_variation)[1]
        for base in training.base_tape_slots
    )
    for offset, case in zip((3, 5), config.test_cases):
        assert case.name == f"extrapolation_{offset}"
        assert case.input_length == input_ceiling + offset
        assert case.tape_slots > tape_ceiling
        assert case.input_length <= case.tape_slots - 1
        assert case.free_steps >= 1 and case.supervision_steps >= 1

    generator = torch.Generator().manual_seed(91)
    for _ in range(100):
        trials = sample_trials(
            training, ("reverse", "copy"), generator, minimum_tape_slots=4
        )
        assert len(trials) == len(training.base_tape_slots)
        for trial in trials:
            assert 4 <= trial.tape_slots <= tape_ceiling
            assert trial.input_max_length <= min(input_ceiling, trial.tape_slots - 1)
            low, high = varied_bounds(
                training.free_steps_per_tape_slot * trial.tape_slots,
                training.time_variation,
            )
            assert low <= trial.free_steps <= high
            assert trial.supervision_steps == round_half_up(
                training.supervision_ratio * trial.free_steps
            )
