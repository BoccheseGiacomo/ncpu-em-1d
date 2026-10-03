from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from .config import ExperimentConfig
from .model import NeuralCellularAutomaton
from .tape import TERNARY_THRESHOLD, TapeLayout, quantize
from .tasks import MultiTaskDataset, binary_tasks
from .training import TrialSpec, sample_trials, supervised_loss


@dataclass(frozen=True)
class ValidationReport:
    checks: tuple[str, ...]
    parameter_count: int
    sample_trials: tuple[TrialSpec, ...]
    task_names: tuple[str, ...]

    def __str__(self) -> str:
        return (
            f"Validated: {', '.join(self.checks)}\n"
            f"Parameters: {self.parameter_count:,}\n"
            f"Tasks: {', '.join(self.task_names)}"
        )


def validate_experiment(
    config: ExperimentConfig, task_names: tuple[str, ...] = ("reverse",)
) -> ValidationReport:
    config.validate()
    trials = sample_trials(
        config.training,
        task_names,
        torch.Generator().manual_seed(config.training.seed),
        minimum_tape_slots=config.model.program_length,
    )
    if len(task_names) > 1 and config.model.program_mode == "zero":
        raise ValueError("multi-task training requires distinct task programs")
    for case in config.test_cases:
        try:
            MultiTaskDataset.from_tasks(
                binary_tasks(task_names, case.input_length, include_shorter=False),
                case.tape_slots,
            )
        except ValueError as error:
            raise ValueError(f"test case {case.name!r}: {error}") from error
    checks = ["sampled trials", "fixed exact-length test cases"]
    values = torch.tensor(
        [
            -TERNARY_THRESHOLD - 1e-6,
            -TERNARY_THRESHOLD,
            0.0,
            TERNARY_THRESHOLD,
            TERNARY_THRESHOLD + 1e-6,
        ]
    )
    if quantize(values).tolist() != [-1, 0, 0, 0, 1]:
        raise AssertionError("ternary thresholds are not strict at 0.333")
    checks.append("direct ternary tape")
    # Deterministic structural probes must not consume training randomness.
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(
            torch.Generator().manual_seed(config.training.seed).get_state()
        )
        model = NeuralCellularAutomaton(config.model, task_names)
        probe_config = replace(config.model, fire_rate=1.0)
        probe = NeuralCellularAutomaton(probe_config, task_names)
        slots = trials[0].tape_slots
        count = len(task_names)
        inputs = torch.zeros(count, slots)
        inputs[:, 0] = 1.0
        indices = torch.arange(count)
        state = probe.initial_state(inputs, indices)
        if not torch.equal(state[:, config.model.io_channel], inputs):
            raise AssertionError("input injection failed")
        program_count = config.model.program_channels
        if not torch.equal(
            state[:, :program_count],
            probe.program_tape(indices, slots),
        ):
            raise AssertionError("program placement failed")
        expected = state - config.model.state_leak * state * probe.update_mask
        if config.model.max_abs_state is not None:
            expected = expected.clamp(
                -config.model.max_abs_state, config.model.max_abs_state
            )
        expected = torch.where(probe.mutable_mask, expected, state)
        torch.testing.assert_close(probe(state, 1)[:, 1], expected)
        checks.append("program placement and masked leak")
        # A nonzero delta projection exposes the program-to-loss gradient path.
        with torch.no_grad():
            probe.rule.hidden.weight.fill_(0.01)
            probe.rule.hidden.bias.fill_(0.1)
            probe.rule.output.weight.fill_(0.01)
        rollout = probe(state, 2)
        if not config.model.program_mutable:
            torch.testing.assert_close(
                rollout[:, :, :program_count],
                state[:, None, :program_count].expand(-1, 3, -1, -1),
                rtol=0,
                atol=0,
            )
        loss = supervised_loss(
            rollout,
            torch.zeros_like(inputs),
            TapeLayout(slots),
            config.model.io_channel,
            0,
            2,
            indices,
            count,
        ).total
        loss.backward()
        if probe.rule.output.weight.grad is None:
            raise AssertionError("rule gradient is missing")
        if config.model.program_mode != "zero":
            gradient = probe.programs.grad
            if gradient is None or not torch.isfinite(gradient).all():
                raise AssertionError("program gradient is missing or non-finite")
        checks.append("full-tape loss and program gradients")
    return ValidationReport(tuple(checks), model.parameter_count, trials, task_names)
