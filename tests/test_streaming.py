import hashlib
import itertools
import json
import tracemalloc

import pytest
import torch

from ncpu_computer_1d import (
    ExperimentConfig,
    MultiTaskDataset,
    NeuralCellularAutomaton,
    StringExample,
    StringTask,
    TaskDataset,
    Trainer,
    binary_strings,
    binary_tasks,
    encode_strings,
    evaluate,
    load_model,
    validate_experiment,
)
from ncpu_computer_1d.evaluation import _sample_indices
from ncpu_computer_1d.tasks import BinaryExamples, output_length_bound
from ncpu_computer_1d.training import _task_signatures
from test_training import tiny_config


def reference_strings(length, shorter=True, empty=True):
    lengths = range(int(not empty), length + 1) if shorter else (length,)
    return tuple(
        "".join(bits) for n in lengths for bits in itertools.product("01", repeat=n)
    )


def reference_target(name, value):
    n = len(value)
    inverse = "".join(str(1 - int(bit)) for bit in value)
    targets = {
        "copy": value,
        "bit_not": inverse,
        "reverse": value[::-1],
        "reverse_not": inverse[::-1],
        "shift_left_zero": value[1:] + "0" if n else "",
        "shift_right_zero": "0" + value[:-1] if n else "",
        "gray_encode": (
            format(int(value, 2) ^ (int(value, 2) >> 1), f"0{n}b") if n else ""
        ),
        "prefix_xor": "".join(
            str(sum(map(int, value[:i])) % 2) for i in range(1, n + 1)
        ),
        "increment": format((int(value, 2) + 1) % (1 << n), f"0{n}b") if n else "",
        "parity": str(sum(map(int, value)) % 2),
        "append_0": value + "0",
        "append_1": value + "1",
    }
    return targets[name]


NAMES = (
    "copy",
    "bit_not",
    "reverse",
    "reverse_not",
    "shift_left_zero",
    "shift_right_zero",
    "gray_encode",
    "prefix_xor",
    "increment",
    "parity",
    "append_0",
    "append_1",
)


@pytest.mark.parametrize("length", range(6))
@pytest.mark.parametrize("shorter", (False, True))
def test_indexed_domains_and_all_tasks_match_independent_exhaustive_reference(
    length, shorter
):
    strings = reference_strings(length, shorter)
    domain = binary_strings(length, include_shorter=shorter, include_empty=shorter)
    assert tuple(domain) == strings
    assert domain[-1] == strings[-1]
    assert domain[::-2] == strings[::-2]
    with pytest.raises(IndexError):
        domain[len(domain)]
    with pytest.raises(IndexError):
        domain[-len(domain) - 1]
    for task in binary_tasks(NAMES, length, include_shorter=shorter):
        expected = tuple(
            StringExample(s, reference_target(task.name, s)) for s in strings
        )
        assert tuple(task.examples) == expected
        assert task.maximum_lengths == (length, max(len(e.target) for e in expected))
        assert output_length_bound(task.name, length) == task.maximum_lengths[1]
        dataset = TaskDataset.from_task(task, length + 2)
        inputs, targets, lengths = dataset.take(torch.arange(len(dataset)))
        torch.testing.assert_close(inputs, encode_strings(strings, length + 2))
        torch.testing.assert_close(
            targets, encode_strings(tuple(e.target for e in expected), length + 2)
        )
        assert lengths.tolist() == [len(e.target) for e in expected]


def test_domain_without_empty_and_addition_order():
    from ncpu_computer_1d import addition_task

    assert tuple(binary_strings(4, include_empty=False)) == reference_strings(
        4, empty=False
    )
    for bits in (1, 2, 3):
        expected = tuple(
            StringExample(f"{a:b}B{b:b}", f"{a+b:b}")
            for a in range(1 << bits)
            for b in range(1 << bits)
        )
        task = addition_task(bits)
        assert tuple(task.examples) == expected
        assert task.maximum_lengths == (
            max(len(e.input) for e in expected),
            max(len(e.target) for e in expected),
        )


def test_training_batches_and_rng_are_identical_to_materialized_sampling():
    strings = reference_strings(4)
    datasets = MultiTaskDataset.from_tasks(binary_tasks(NAMES, 4), 6)
    actual_rng = torch.Generator().manual_seed(43)
    reference_rng = torch.Generator().manual_seed(43)
    for _ in range(4):
        batches = []
        for task_id, name in enumerate(NAMES):
            targets = tuple(reference_target(name, s) for s in strings)
            indices = torch.randint(len(strings), (7,), generator=reference_rng)
            batches.append(
                (
                    encode_strings(strings, 6)[indices],
                    encode_strings(targets, 6)[indices],
                    torch.tensor(list(map(len, targets)))[indices],
                    torch.full((7,), task_id, dtype=torch.int64),
                )
            )
        combined = tuple(torch.cat(values) for values in zip(*batches))
        order = torch.randperm(len(NAMES) * 7, generator=reference_rng)
        actual = datasets.balanced_sample(7, actual_rng)
        for a, b in zip(actual, combined):
            torch.testing.assert_close(a, b[order], rtol=0, atol=0)
        assert torch.equal(actual_rng.get_state(), reference_rng.get_state())


def test_explicit_custom_multitask_inputs_compare_without_targets():
    tasks = (
        StringTask("a", (StringExample("1B0", "1"), StringExample("", ""))),
        StringTask(
            "b", (StringExample("1B0", "0B1"), StringExample("", "")), "multiple"
        ),
    )
    datasets = MultiTaskDataset.from_tasks(tasks, 5)
    assert len(datasets) == 4
    with pytest.raises(ValueError, match="ordered inputs"):
        MultiTaskDataset.from_tasks(
            (tasks[0], StringTask("c", (StringExample("0", "0"),))), 5
        )


def test_large_domains_do_not_enumerate_in_checks_or_fingerprints(monkeypatch):
    config = ExperimentConfig()

    def forbid_enumeration(self, index):
        raise AssertionError("setup must not access examples")

    with monkeypatch.context() as guarded:
        guarded.setattr(BinaryExamples, "__getitem__", forbid_enumeration)
        validate_experiment(config, ("reverse", "copy"))
        assert len(_task_signatures(config, ("reverse", "copy"))) == 2
        trainer = Trainer(config, (("reverse", 2.0), ("copy", 1.0)))
        assert trainer.task_signatures
    tracemalloc.start()
    try:
        for length in (16, 19, 21, 62):
            datasets = MultiTaskDataset.from_tasks(
                binary_tasks(NAMES, length), length + 2
            )
            assert len(datasets.datasets[0]) == (1 << (length + 1)) - 1
            for dataset in datasets.datasets:
                assert len(dataset.signature) == 64
                batch = dataset.take(torch.tensor([0, 1, len(dataset) - 1]))
                assert batch[0].shape == batch[1].shape == (3, length + 2)
                assert batch[2].shape == (3,)
                assert set(vars(dataset)) == {"task", "tape_slots"}
        assert tracemalloc.get_traced_memory()[1] < 1_000_000
    finally:
        tracemalloc.stop()


def test_streaming_evaluation_matches_eager_full_state_reference(monkeypatch):
    trainer = Trainer(tiny_config(), (("reverse", 2.0),))
    trainer.train_step()
    task = binary_tasks(("reverse",), 3, include_shorter=False)[0]
    indexed = TaskDataset.from_task(task, 6)
    explicit = TaskDataset.from_task(StringTask(task.name, tuple(task.examples)), 6)
    visited = []
    original_take = TaskDataset.take

    def record_take(self, indices):
        visited.extend(indices.tolist())
        assert len(indices) <= 3
        return original_take(self, indices)

    with monkeypatch.context() as guarded:
        guarded.setattr(TaskDataset, "take", record_take)
        result = evaluate(
            trainer.model, indexed, steps=7, step_start=4, step_end=7, batch_size=3
        )
    assert visited == list(range(8))
    forward = trainer.model.forward

    def full_state_reference(state, steps, **kwargs):
        assert kwargs.pop("io_only")
        return forward(state, steps, **kwargs)[:, :, trainer.model.config.io_channel]

    monkeypatch.setattr(trainer.model, "forward", full_state_reference)
    reference = evaluate(
        trainer.model, explicit, steps=7, step_start=4, step_end=7, batch_size=3
    )
    assert result == reference


def test_subset_is_bounded_unique_reproducible_and_shared_across_tasks(monkeypatch):
    original_randperm = torch.randperm

    def bounded_randperm(count, **kwargs):
        assert count <= 7
        return original_randperm(count, **kwargs)

    monkeypatch.setattr(torch, "randperm", bounded_randperm)
    first = _sample_indices(1 << 21, 7, 4)
    assert len(set(first.tolist())) == 7
    assert torch.equal(first, _sample_indices(1 << 21, 7, 4))
    assert not torch.equal(first, _sample_indices(1 << 21, 7, 5))
    assert set(_sample_indices(8, 7, 0).tolist()) <= set(range(8))
    model = NeuralCellularAutomaton(tiny_config().model, ("copy", "reverse"))
    datasets = MultiTaskDataset.from_tasks(
        binary_tasks(model.task_names, 21, include_shorter=False), 27
    )
    observed = []
    take = TaskDataset.take

    def record(self, indices):
        observed.append(indices.tolist())
        return take(self, indices)

    monkeypatch.setattr(TaskDataset, "take", record)
    results = [
        evaluate(
            model, d, steps=2, step_start=1, step_end=2, max_examples=7, batch_size=3
        )
        for d in datasets.datasets
    ]
    assert observed[:3] == observed[3:]
    assert all(r.examples == 7 and r.total_examples == 1 << 21 for r in results)


def test_subset_algorithm_is_exactly_uniform_on_a_small_population(monkeypatch):
    from collections import Counter

    counts = Counter()
    monkeypatch.setattr(torch, "randperm", lambda count, **kwargs: torch.arange(count))
    for draws in itertools.product(range(3), range(4)):
        stream = iter(draws)
        monkeypatch.setattr(
            torch, "randint", lambda *args, **kwargs: torch.tensor(next(stream))
        )
        counts[frozenset(_sample_indices(4, 2, 0).tolist())] += 1
    assert len(counts) == 6
    assert set(counts.values()) == {2}


def test_validation_streams_all_lengths_and_partial_batches(monkeypatch):
    trainer = Trainer(tiny_config(), (("reverse", 2.0), ("copy", 1.0)))
    observed = {}
    take = TaskDataset.take

    def record(self, indices):
        key = self.name, self.tape_slots
        observed.setdefault(key, []).extend(indices.tolist())
        assert len(indices) <= trainer.config.training.batch_size_per_task
        return take(self, indices)

    monkeypatch.setattr(TaskDataset, "take", record)
    trainer.validation_metrics()
    assert observed == {
        (name, slots): list(range((1 << (maximum + 1)) - 1))
        for name in trainer.task_names
        for slots, maximum in ((4, 1), (6, 3))
    }


def legacy_hash(dataset):
    data = {
        "name": dataset.name,
        "output_mode": dataset.output_mode,
        "inputs": tuple(dataset.input_strings),
        "targets": tuple(dataset.target_strings),
    }
    return hashlib.sha256(
        json.dumps(data, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def test_streamed_legacy_hash_matches_original_json():
    for task in binary_tasks(NAMES, 3):
        dataset = TaskDataset.from_task(task, 5)
        assert dataset.legacy_signature() == legacy_hash(dataset)
    custom = TaskDataset.from_task(
        StringTask("unicode-\u00e8", (StringExample("1B0", "0"),)), 4
    )
    assert custom.signature == legacy_hash(custom)


def test_legacy_checkpoint_resume_is_exact_and_migrates_signatures(tmp_path):
    config = tiny_config()
    specs = (("reverse", 2.0), ("copy", 1.0))
    trainer = Trainer(config, specs)
    trainer.train_step()
    checkpoint = trainer.checkpoint()
    checkpoint.pop("task_signature_format")
    # Old signature covers the maximum feasible sampled input domain (length 4).
    old_datasets = MultiTaskDataset.from_tasks(binary_tasks(trainer.task_names, 4), 7)
    checkpoint["task_signatures"] = tuple(legacy_hash(d) for d in old_datasets.datasets)
    path = tmp_path / "legacy.pt"
    torch.save(checkpoint, path)
    with torch.random.fork_rng(devices=[]):
        load_model(path, "cpu")
    expected = trainer.train_step()
    weights = {key: value.clone() for key, value in trainer.model.state_dict().items()}
    restored = Trainer.from_checkpoint(path)
    actual = restored.train_step()
    assert actual.to_dict() == expected.to_dict()
    for key, value in weights.items():
        torch.testing.assert_close(
            value, restored.model.state_dict()[key], rtol=0, atol=0
        )
    assert restored.checkpoint()["task_signature_format"] == "indexed-v1"
    assert restored.task_signatures != checkpoint["task_signatures"]
    checkpoint["task_signatures"] = ("wrong", "wrong")
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="task definitions"):
        Trainer.from_checkpoint(path)


def test_invalid_domains_and_empty_batch():
    for invalid in (-1, 63, 2.5, True):
        with pytest.raises(ValueError):
            binary_strings(invalid)
    dataset = TaskDataset.from_task(binary_tasks(("reverse",), 1)[0], 3)
    assert [
        tuple(t.shape) for t in dataset.take(torch.empty(0, dtype=torch.int64))
    ] == [(0, 3), (0, 3), (0,)]
    with pytest.raises(ValueError, match="one-dimensional"):
        dataset.take(torch.zeros(1, 1, dtype=torch.int64))
    with pytest.raises(IndexError):
        dataset.take(torch.tensor([len(dataset)]))
