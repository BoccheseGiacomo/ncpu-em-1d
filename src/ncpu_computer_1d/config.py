from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any


SUPPORTED_GATES = {"none", "linear", "sigmoid", "tanh", "relu"}
SUPPORTED_ACTIVATIONS = {"relu", "softplus"}
PROGRAM_MODES = {"zero", "learned_read_only", "learned_mutable"}
LR_INTERPOLATIONS = {"linear", "cosine"}


@dataclass(frozen=True)
class ModelConfig:
    program_channels: int = 4
    computation_channels: int = 11
    program_mode: str = "learned_read_only"
    program_length: int = 1
    program_placement: str = "periodic"
    program_init_std: float = 0.1
    hidden_size: int = 96
    radius: int = 1
    learnable_kernels: int | None = None
    learnable_kernel_init: str = "laplacian"
    activation: str = "relu"
    gate: str = "none"
    gate_bias: float = 1.0
    fire_rate: float = 0.9
    state_leak: float = 0.02
    max_abs_state: float | None = 10.0
    random_kernel_seed: int = 0

    @property
    def fixed_kernels(self) -> tuple[str, ...]:
        names = ("identity", "sobel_x", "laplacian", "neighbor_mean")
        if self.radius == 2:
            names += ("outer_derivative", "neighbor_contrast")
        return names

    @property
    def learned_kernel_count(self) -> int:
        return self.radius if self.learnable_kernels is None else self.learnable_kernels

    @property
    def io_channel(self) -> int:
        return self.program_channels

    @property
    def channels(self) -> int:
        return self.program_channels + 1 + self.computation_channels

    @property
    def program_mutable(self) -> bool:
        return self.program_mode == "learned_mutable"

    def validate(self) -> None:
        integers = (
            self.program_channels,
            self.program_length,
            self.computation_channels,
            self.hidden_size,
            self.radius,
            self.random_kernel_seed,
        )
        if any(type(value) is not int for value in integers):
            raise TypeError("model dimensions, channels, and seeds must be integers")
        if self.radius not in {1, 2}:
            raise ValueError("radius must be 1 or 2")
        if self.program_channels < 1:
            raise ValueError("program_channels must be positive")
        if self.program_length < 1:
            raise ValueError("program_length must be positive")
        if self.program_placement not in {"periodic", "prefix"}:
            raise ValueError("program_placement must be 'periodic' or 'prefix'")
        if self.computation_channels < 0:
            raise ValueError("computation_channels cannot be negative")
        if self.program_mode not in PROGRAM_MODES:
            raise ValueError(f"program_mode must be one of {sorted(PROGRAM_MODES)}")
        if self.hidden_size < 1:
            raise ValueError("hidden_size must be positive")
        if self.random_kernel_seed < 0:
            raise ValueError("random_kernel_seed cannot be negative")
        numeric = (
            self.program_init_std,
            self.gate_bias,
            self.fire_rate,
            self.state_leak,
        )
        if self.max_abs_state is not None:
            numeric += (self.max_abs_state,)
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("model scalar hyperparameters must be finite")
        if self.program_init_std < 0:
            raise ValueError("program_init_std cannot be negative")
        if self.learnable_kernels is not None and (
            type(self.learnable_kernels) is not int or self.learnable_kernels < 0
        ):
            raise ValueError("learnable_kernels must be a non-negative integer or None")
        if self.learnable_kernel_init not in {"laplacian", "random"}:
            raise ValueError("learnable_kernel_init must be 'laplacian' or 'random'")
        if self.activation not in SUPPORTED_ACTIVATIONS:
            raise ValueError(
                f"activation must be one of {sorted(SUPPORTED_ACTIVATIONS)}"
            )
        if self.gate not in SUPPORTED_GATES:
            raise ValueError(f"gate must be one of {sorted(SUPPORTED_GATES)}")
        if not 0.0 < self.fire_rate <= 1.0:
            raise ValueError("fire_rate must be in (0, 1]")
        if not 0.0 <= self.state_leak <= 1.0:
            raise ValueError("state_leak must be in [0, 1]")
        if self.max_abs_state is not None and self.max_abs_state <= 0:
            raise ValueError("max_abs_state must be positive or None")


@dataclass(frozen=True)
class TestCase:
    __test__ = False

    name: str
    tape_slots: int
    input_length: int
    free_steps: int
    supervision_steps: int

    def validate(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("test case name must be a non-empty string")
        values = (
            self.tape_slots,
            self.input_length,
            self.free_steps,
            self.supervision_steps,
        )
        if any(type(value) is not int for value in values):
            raise TypeError("test case sizes and times must be integers")
        if self.tape_slots < 2:
            raise ValueError("test tape must have at least two slots")
        if not 0 <= self.input_length <= self.tape_slots - 1:
            raise ValueError("test input_length must be in [0, tape_slots - 1]")
        if self.free_steps < 0 or self.supervision_steps < 1:
            raise ValueError("test evolution times are invalid")

    @property
    def steps(self) -> int:
        return self.free_steps + self.supervision_steps


@dataclass(frozen=True)
class TrainingConfig:
    updates: int = 2000
    batch_size_per_task: int = 64
    base_tape_slots: tuple[int, ...] = (5, 8, 11, 15, 20)
    base_input_max_lengths: tuple[int, ...] = (3, 5, 7, 10, 13)
    tape_variation: float = 0.2
    input_variation: float = 0.2
    free_steps_per_tape_slot: float = 3.0
    time_variation: float = 0.2
    supervision_ratio: float = 1.5
    lr_points: tuple[tuple[float, float], ...] = (
        (0.0, 2e-3),
        (0.35, 2e-3),
        (0.60, 7e-4),
        (0.75, 7e-4),
        (0.95, 1e-4),
        (1.0, 1e-4),
    )
    lr_interpolation: str = "cosine"
    weight_decay: float = 2e-5
    program_weight_decay: float = 1e-4
    grad_clip: float | None = 0.8
    train_rule: bool = True
    train_program: bool = True
    perception_noise_start: float = 0.0
    perception_noise_end: float = 0.0
    seed: int = 0
    validation_every: int = 50
    checkpoint_every: int = 50
    device: str = "auto"

    def validate(self) -> None:
        integers = (
            self.updates,
            self.batch_size_per_task,
            self.seed,
            self.validation_every,
            self.checkpoint_every,
        )
        if any(type(value) is not int for value in integers):
            raise TypeError("training counts, ranges, and seed must be integers")
        if any(
            type(value) is not bool for value in (self.train_rule, self.train_program)
        ):
            raise TypeError("training flags must be booleans")
        numeric = (
            self.tape_variation,
            self.input_variation,
            self.free_steps_per_tape_slot,
            self.time_variation,
            self.supervision_ratio,
            self.weight_decay,
            self.program_weight_decay,
            self.perception_noise_start,
            self.perception_noise_end,
        )
        if self.grad_clip is not None:
            numeric += (self.grad_clip,)
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("training scalar hyperparameters must be finite")
        if self.updates < 1 or self.batch_size_per_task < 1:
            raise ValueError("updates and batch size must be positive")
        if not self.base_tape_slots or (
            len(self.base_tape_slots) != len(self.base_input_max_lengths)
        ):
            raise ValueError(
                "base tape and input tuples must have equal nonzero length"
            )
        if any(type(value) is not int for value in self.base_tape_slots):
            raise TypeError("base tape slots must be integers")
        if any(type(value) is not int for value in self.base_input_max_lengths):
            raise TypeError("base input lengths must be integers")
        if any(value < 2 for value in self.base_tape_slots):
            raise ValueError("base tape slots must be at least two")
        if any(value < 0 for value in self.base_input_max_lengths):
            raise ValueError("base input lengths cannot be negative")
        if len(set(self.base_tape_slots)) != len(self.base_tape_slots):
            raise ValueError("base tape slots must be unique")
        if len(set(self.base_input_max_lengths)) != len(self.base_input_max_lengths):
            raise ValueError("base input lengths must be unique")
        variations = (
            self.tape_variation,
            self.input_variation,
            self.time_variation,
        )
        if any(not 0 <= value < 1 for value in variations):
            raise ValueError("variations must be in [0, 1)")
        if self.free_steps_per_tape_slot <= 0:
            raise ValueError("free_steps_per_tape_slot must be positive")
        if self.supervision_ratio <= 0:
            raise ValueError("supervision_ratio must be positive")
        self._validate_lr_schedule()
        if self.weight_decay < 0 or self.program_weight_decay < 0:
            raise ValueError("weight decays cannot be negative")
        if self.grad_clip is not None and self.grad_clip <= 0:
            raise ValueError("grad_clip must be positive or None")
        if self.perception_noise_start < 0 or self.perception_noise_end < 0:
            raise ValueError("perception noise cannot be negative")
        if self.seed < 0:
            raise ValueError("seed cannot be negative")
        if self.validation_every < 1 or self.checkpoint_every < 1:
            raise ValueError("validation and checkpoint intervals must be positive")
        if self.device != "auto" and not (
            self.device == "cpu" or self.device.startswith("cuda")
        ):
            raise ValueError("device must be 'auto', 'cpu', or a CUDA device")

    def _validate_lr_schedule(self) -> None:
        if self.lr_interpolation not in LR_INTERPOLATIONS:
            raise ValueError(
                f"lr_interpolation must be one of {sorted(LR_INTERPOLATIONS)}"
            )
        if len(self.lr_points) < 2:
            raise ValueError("lr_points must contain at least two anchors")
        points = []
        for point in self.lr_points:
            if not isinstance(point, (tuple, list)) or len(point) != 2:
                raise TypeError("each learning-rate anchor must be a pair")
            progress, rate = point
            if any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                for value in point
            ):
                raise TypeError("learning-rate anchors must be numeric")
            if not math.isfinite(progress) or not math.isfinite(rate):
                raise ValueError("learning-rate anchors must be finite")
            if rate <= 0:
                raise ValueError("learning rates must be positive")
            points.append((float(progress), float(rate)))
        if points[0][0] != 0.0 or points[-1][0] != 1.0:
            raise ValueError("lr_points must start at 0.0 and end at 1.0")
        if any(right[0] <= left[0] for left, right in zip(points, points[1:])):
            raise ValueError("learning-rate progress values must strictly increase")


@dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    test_cases: tuple[TestCase, ...] = (
        TestCase("extrapolation_3", 25, 19, 75, 113),
        TestCase("extrapolation_5", 27, 21, 81, 122),
    )

    def validate(self) -> None:
        self.model.validate()
        self.training.validate()
        if not self.test_cases:
            raise ValueError("at least one test case is required")
        for case in self.test_cases:
            case.validate()
            if case.tape_slots < self.model.program_length:
                raise ValueError(f"test case {case.name!r} cannot contain the program")
        if len({case.name for case in self.test_cases}) != len(self.test_cases):
            raise ValueError("test case names must be unique")
        if not self.training.train_rule and not self.training.train_program:
            raise ValueError(
                "at least one of train_rule and train_program must be true"
            )
        if self.training.train_program and self.model.program_mode == "zero":
            raise ValueError("a zero program cannot be trained")
        _validate_trial_feasibility(self.training, self.model.program_length)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentConfig":
        model_data = dict(data["model"])
        training_data = dict(data["training"])
        training_data["base_tape_slots"] = tuple(training_data["base_tape_slots"])
        training_data["base_input_max_lengths"] = tuple(
            training_data["base_input_max_lengths"]
        )
        training_data["lr_points"] = tuple(
            tuple(point) for point in training_data["lr_points"]
        )
        config = cls(
            model=ModelConfig(**model_data),
            training=TrainingConfig(**training_data),
            test_cases=tuple(TestCase(**case) for case in data["test_cases"]),
        )
        config.validate()
        return config


def round_half_up(value: float) -> int:
    if not math.isfinite(value) or value < 0:
        raise ValueError("round_half_up expects a finite non-negative value")
    return math.floor(value + 0.5)


def varied_bounds(base: int, variation: float) -> tuple[int, int]:
    return (
        round_half_up(base * (1.0 - variation)),
        round_half_up(base * (1.0 + variation)),
    )


def tape_bounds(
    training: TrainingConfig, base_tape: int, minimum: int = 1
) -> tuple[int, int]:
    low, high = varied_bounds(base_tape, training.tape_variation)
    low = max(low, minimum)
    if low > high:
        raise ValueError(
            f"tape range for base {base_tape} cannot contain program length {minimum}"
        )
    return low, high


def _validate_trial_feasibility(training: TrainingConfig, minimum: int) -> None:
    for base_tape, base_input in zip(
        training.base_tape_slots, training.base_input_max_lengths
    ):
        if base_input > max(base_tape, minimum) - 1:
            raise ValueError("every base input must leave one blank tape cell")
        tape_low, _ = tape_bounds(training, base_tape, minimum)
        input_low, _ = varied_bounds(base_input, training.input_variation)
        if tape_low < 2:
            raise ValueError("tape variation can produce fewer than two slots")
        if input_low > tape_low - 1:
            raise ValueError(
                "input variation has no feasible value for every sampled tape"
            )
        minimum_steps = round_half_up(
            training.free_steps_per_tape_slot
            * tape_low
            * (1.0 - training.time_variation)
        )
        if minimum_steps < 1:
            raise ValueError("time variation can produce zero free steps")
        if round_half_up(training.supervision_ratio * minimum_steps) < 1:
            raise ValueError("supervision_ratio can produce an empty window")
