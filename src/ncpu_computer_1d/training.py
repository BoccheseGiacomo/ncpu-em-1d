from __future__ import annotations

import math
import random
import shutil
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Callable

import torch

from .config import (
    ExperimentConfig,
    TrainingConfig,
    round_half_up,
    tape_bounds,
    varied_bounds,
)
from .model import NeuralCellularAutomaton
from .tape import TapeLayout, quantize
from .tasks import MultiTaskDataset, StringTask, binary_tasks, semantic_correct


CHECKPOINT_FORMAT = "ncpu-computer-1d-v2"


def validate_task_specs(
    task_specs: tuple[tuple[str, float], ...],
) -> tuple[tuple[str, ...], tuple[float, ...]]:
    if not task_specs:
        raise ValueError("task specs must not be empty")
    names = []
    weights = []
    for spec in task_specs:
        if not isinstance(spec, tuple) or len(spec) != 2:
            raise ValueError("each task spec must be a (name, weight) tuple")
        name, weight = spec
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise ValueError("task weights must be numbers")
        weight = float(weight)
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError("task weights must be finite and positive")
        names.append(name)
        weights.append(weight)
    task_names = tuple(names)
    binary_tasks(task_names, 0)
    return task_names, tuple(weights)


def _tasks_fit_tape(tasks: tuple[StringTask, ...], tape_slots: int) -> bool:
    maximum_length = tape_slots - 1
    return all(max(task.maximum_lengths) <= maximum_length for task in tasks)


@lru_cache(maxsize=None)
def _training_pair_fits(
    task_names: tuple[str, ...], input_max_length: int, tape_slots: int
) -> bool:
    return _tasks_fit_tape(binary_tasks(task_names, input_max_length), tape_slots)


def _input_bounds(training: TrainingConfig, base_input: int) -> range:
    low, high = varied_bounds(base_input, training.input_variation)
    return range(low, high + 1)


@lru_cache(maxsize=128)
def _validate_task_feasibility(
    training: TrainingConfig, task_names: tuple[str, ...], minimum_tape_slots: int = 1
) -> None:
    training.validate()
    binary_tasks(task_names, 0)
    for base_tape, base_input in zip(
        training.base_tape_slots, training.base_input_max_lengths
    ):
        if not _training_pair_fits(
            task_names, base_input, max(base_tape, minimum_tape_slots)
        ):
            raise ValueError(
                f"base training/validation pair tape={base_tape}, "
                f"input={base_input} does not leave one blank input and output "
                "cell for every task"
            )
        tape_low, tape_high = tape_bounds(training, base_tape, minimum_tape_slots)
        candidates = _input_bounds(training, base_input)
        for tape_slots in range(tape_low, tape_high + 1):
            if not any(
                _training_pair_fits(task_names, input_max, tape_slots)
                for input_max in candidates
            ):
                raise ValueError(
                    f"sampled tape capacity {tape_slots} has no feasible input "
                    "length that leaves one blank input and output cell for "
                    "every task"
                )


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _task_signatures(
    config: ExperimentConfig, task_names: tuple[str, ...], *, legacy: bool = False
) -> tuple[str, ...]:
    _validate_task_feasibility(config.training, task_names, config.model.program_length)
    tape_ranges = tuple(
        tape_bounds(config.training, base, config.model.program_length)
        for base in config.training.base_tape_slots
    )
    input_bounds = tuple(
        varied_bounds(base, config.training.input_variation)
        for base in config.training.base_input_max_lengths
    )
    feasible_inputs = [
        input_max
        for (tape_low, tape_high), (input_low, input_high) in zip(
            tape_ranges, input_bounds
        )
        for tape_slots in range(tape_low, tape_high + 1)
        for input_max in range(input_low, input_high + 1)
        if _training_pair_fits(task_names, input_max, tape_slots)
    ]
    maximum_input = max(feasible_inputs)
    maximum_tape = max(high for _, high in tape_ranges)
    datasets = MultiTaskDataset.from_tasks(
        binary_tasks(task_names, maximum_input),
        maximum_tape,
    )
    if legacy:
        return tuple(dataset.legacy_signature() for dataset in datasets.datasets)
    return datasets.signatures


@dataclass(frozen=True)
class TrialSpec:
    tape_slots: int
    input_max_length: int
    free_steps: int
    supervision_steps: int

    @property
    def steps(self) -> int:
        return self.free_steps + self.supervision_steps


def _sample_varied(
    base: int | float, variation: float, generator: torch.Generator, minimum: int = 0
) -> int:
    draw = float(torch.rand((), generator=generator))
    if minimum > round_half_up(base * (1.0 - variation)):
        low = minimum - 0.5
        high = base * (1.0 + variation)
        return round_half_up(low + draw * (high - low))
    delta = (2.0 * draw - 1.0) * variation
    return round_half_up(base * (1.0 + delta))


def sample_trials(
    training: TrainingConfig,
    task_names: tuple[str, ...],
    generator: torch.Generator,
    *,
    minimum_tape_slots: int = 1,
) -> tuple[TrialSpec, ...]:
    if type(minimum_tape_slots) is not int or minimum_tape_slots < 1:
        raise ValueError("minimum_tape_slots must be positive")
    _validate_task_feasibility(training, task_names, minimum_tape_slots)
    trials = []
    for base_tape, base_input in zip(
        training.base_tape_slots, training.base_input_max_lengths
    ):
        tape_slots = _sample_varied(
            base_tape, training.tape_variation, generator, minimum_tape_slots
        )
        input_max = _sample_varied(base_input, training.input_variation, generator)
        while not _training_pair_fits(task_names, input_max, tape_slots):
            input_max = _sample_varied(base_input, training.input_variation, generator)
        free_steps = _sample_varied(
            training.free_steps_per_tape_slot * tape_slots,
            training.time_variation,
            generator,
        )
        supervision_steps = round_half_up(training.supervision_ratio * free_steps)
        trials.append(TrialSpec(tape_slots, input_max, free_steps, supervision_steps))
    order = torch.randperm(len(trials), generator=generator).tolist()
    return tuple(trials[index] for index in order)


def validation_trials(
    training: TrainingConfig,
    task_names: tuple[str, ...],
    *,
    minimum_tape_slots: int = 1,
) -> tuple[TrialSpec, ...]:
    _validate_task_feasibility(training, task_names, minimum_tape_slots)
    trials = []
    for tape_slots, input_max in zip(
        training.base_tape_slots, training.base_input_max_lengths
    ):
        tape_slots = max(tape_slots, minimum_tape_slots)
        free_steps = round_half_up(training.free_steps_per_tape_slot * tape_slots)
        trials.append(
            TrialSpec(
                tape_slots,
                input_max,
                free_steps,
                round_half_up(training.supervision_ratio * free_steps),
            )
        )
    return tuple(trials)


@dataclass(frozen=True)
class LossComponents:
    total: torch.Tensor
    base: torch.Tensor
    per_task: torch.Tensor


def supervised_loss(
    rollout: torch.Tensor,
    target: torch.Tensor,
    layout: TapeLayout,
    io_channel: int,
    free_steps: int,
    supervision_steps: int,
    task_indices: torch.Tensor | None = None,
    task_count: int | None = None,
    task_weights: tuple[float, ...] | None = None,
    *,
    start_step: int = 0,
) -> LossComponents:
    if (
        type(free_steps) is not int
        or free_steps < 0
        or type(supervision_steps) is not int
        or supervision_steps < 1
    ):
        raise ValueError(
            "supervision requires non-negative free steps and positive window"
        )
    if type(start_step) is not int or not 0 <= start_step <= free_steps + 1:
        raise ValueError("start_step cannot skip supervised states")
    start = free_steps + 1 - start_step
    end = start + supervision_steps
    if rollout.ndim not in {3, 4}:
        raise ValueError("rollout must be a full-state or I/O-only trajectory")
    if rollout.ndim == 4 and not 0 <= io_channel < rollout.shape[2]:
        raise ValueError("io_channel does not exist in rollout")
    if end > rollout.shape[1]:
        raise ValueError("rollout does not cover the supervision window")
    if target.shape != (rollout.shape[0], layout.tape_slots):
        raise ValueError("target shape does not match rollout and tape layout")
    values = rollout[:, :, io_channel] if rollout.ndim == 4 else rollout
    prediction = layout.extract_tape(values[:, start:end])
    per_example = (
        (prediction - target.to(prediction.device).unsqueeze(1))
        .square()
        .mean(dim=(1, 2))
    )
    if task_indices is None:
        if task_count is not None or task_weights is not None:
            raise ValueError("task_count and task_weights require task_indices")
        per_task = per_example.mean().reshape(1)
    else:
        task_indices = torch.as_tensor(task_indices, dtype=torch.int64)
        if task_indices.shape != (rollout.shape[0],):
            raise ValueError("task_indices must contain one index per example")
        if type(task_count) is not int or task_count < 1:
            raise ValueError("task_count must be a positive integer")
        if bool(((task_indices < 0) | (task_indices >= task_count)).any()):
            raise ValueError("task index is out of range")
        membership = task_indices[:, None] == torch.arange(
            task_count, device=task_indices.device
        )
        counts = membership.sum(dim=0)
        if bool((counts == 0).any()):
            raise ValueError("every task must occur in a balanced batch")
        membership = membership.to(device=prediction.device, dtype=per_example.dtype)
        per_task = (per_example[:, None] * membership).sum(0) / counts.to(
            prediction.device
        )
    if task_weights is None:
        base = per_task.mean()
    else:
        if len(task_weights) != len(per_task) or any(
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(weight)
            or weight <= 0
            for weight in task_weights
        ):
            raise ValueError("task weights must be finite and positive, one per task")
        weights = torch.as_tensor(task_weights, dtype=per_task.dtype)
        if not bool(torch.isfinite(weights).all()) or bool((weights <= 0).any()):
            raise ValueError(
                "task weights must be finite and positive in the loss dtype"
            )
        weights = weights.to(per_task.device)
        base = (per_task * weights).sum() / weights.sum()
    return LossComponents(base, base, per_task)


def learning_rate_at(training: TrainingConfig, update: int) -> float:
    if not 0 <= update < training.updates:
        raise ValueError("update must be in [0, updates)")
    progress = 0.0 if training.updates == 1 else update / (training.updates - 1)
    for (start, start_rate), (end, end_rate) in zip(
        training.lr_points, training.lr_points[1:]
    ):
        if progress <= end:
            fraction = (progress - start) / (end - start)
            if training.lr_interpolation == "cosine":
                fraction = 0.5 * (1.0 - math.cos(math.pi * fraction))
            return start_rate + (end_rate - start_rate) * fraction
    raise RuntimeError("learning-rate schedule does not cover training progress")


def perception_noise(config: TrainingConfig, update: int) -> float:
    progress = 0.0 if config.updates == 1 else update / (config.updates - 1)
    return config.perception_noise_start + progress * (
        config.perception_noise_end - config.perception_noise_start
    )


@dataclass(frozen=True)
class StepMetrics:
    update: int
    loss: float
    learning_rate: float
    perception_noise: float
    gradient_norm: float
    semantic_accuracy: float
    raw_accuracy: float
    trials: tuple[TrialSpec, ...]
    validation_loss: float | None = None
    validation_accuracies: dict[str, float] | None = None

    def to_dict(self) -> dict:
        data = self.__dict__.copy()
        data["trials"] = [asdict(trial) for trial in self.trials]
        if self.validation_accuracies is not None:
            data["validation_accuracies"] = dict(self.validation_accuracies)
        return data


@dataclass(frozen=True)
class ValidationMetrics:
    loss: float
    task_losses: dict[str, float]
    task_accuracies: dict[str, float]


@dataclass(frozen=True)
class SeedResult:
    seed: int
    checkpoint: Path
    best_validation_loss: float
    history: list[dict]


class Trainer:
    def __init__(
        self,
        config: ExperimentConfig,
        task_specs: tuple[tuple[str, float], ...],
        model: NeuralCellularAutomaton | None = None,
    ):
        config.validate()
        task_names, task_weights = validate_task_specs(task_specs)
        self.config = config
        self.task_specs = tuple(
            (name, weight) for name, weight in zip(task_names, task_weights)
        )
        self.task_names = task_names
        self.task_weights = task_weights
        self.task_signatures = _task_signatures(config, self.task_names)
        if len(task_names) > 1 and config.model.program_mode == "zero":
            raise ValueError("multi-task training requires distinct task programs")
        self.device = resolve_device(config.training.device)
        random.seed(config.training.seed)
        torch.manual_seed(config.training.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.training.seed)
        self.model = (
            NeuralCellularAutomaton(config.model, self.task_names)
            if model is None
            else model
        ).to(self.device)
        if self.model.config != config.model:
            raise ValueError("model and experiment configurations differ")
        if self.model.task_names != self.task_names:
            raise ValueError("model and trainer task order differs")
        for parameter in self.model.shared_parameters:
            parameter.requires_grad_(config.training.train_rule)
        for parameter in self.model.program_parameters:
            parameter.requires_grad_(config.training.train_program)
        groups = []
        if config.training.train_rule:
            groups.append(
                {
                    "params": self.model.shared_parameters,
                    "weight_decay": config.training.weight_decay,
                    "name": "rule",
                }
            )
        if config.training.train_program:
            groups.append(
                {
                    "params": self.model.program_parameters,
                    "weight_decay": config.training.program_weight_decay,
                    "name": "program",
                }
            )
        self.optimizer = torch.optim.Adam(
            groups, lr=learning_rate_at(config.training, 0)
        )
        self.trial_generator = torch.Generator().manual_seed(config.training.seed)
        self.data_generator = torch.Generator().manual_seed(config.training.seed + 1)
        self.current_update = 0
        self.best_validation_loss = math.inf
        self.history: list[dict] = []

    def _datasets(
        self, trial: TrialSpec, include_shorter: bool = True
    ) -> MultiTaskDataset:
        return MultiTaskDataset.from_tasks(
            binary_tasks(
                self.task_names,
                trial.input_max_length,
                include_shorter=include_shorter,
            ),
            trial.tape_slots,
        )

    def _prepare(self, datasets: MultiTaskDataset) -> tuple[torch.Tensor, ...]:
        batch = datasets.balanced_sample(
            self.config.training.batch_size_per_task, self.data_generator
        )
        return (*(value.to(self.device) for value in batch[:-1]), batch[-1])

    def train_step(self) -> StepMetrics:
        self.model.train()
        learning_rate = learning_rate_at(self.config.training, self.current_update)
        noise = perception_noise(self.config.training, self.current_update)
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate
        trials = sample_trials(
            self.config.training,
            self.task_names,
            self.trial_generator,
            minimum_tape_slots=self.config.model.program_length,
        )
        self.optimizer.zero_grad(set_to_none=True)
        totals = torch.zeros(3, device=self.device)
        for trial in trials:
            datasets = self._datasets(trial)
            inputs, targets, lengths, task_indices = self._prepare(datasets)
            layout = TapeLayout(trial.tape_slots)
            initial = self.model.initial_state(inputs, task_indices)
            rollout = self.model(
                initial,
                trial.steps,
                noise,
                io_only=True,
                start_step=trial.free_steps + 1,
            )
            losses = supervised_loss(
                rollout,
                targets,
                layout,
                self.config.model.io_channel,
                trial.free_steps,
                trial.supervision_steps,
                task_indices,
                len(self.task_names),
                self.task_weights,
                start_step=trial.free_steps + 1,
            )
            if not torch.isfinite(losses.total):
                raise FloatingPointError(
                    f"non-finite loss at update {self.current_update + 1}"
                )
            (losses.total / len(trials)).backward()
            totals[0] += losses.total.detach() / len(trials)
            with torch.no_grad():
                final = quantize(layout.extract_tape(rollout[:, -1]))
                expected = targets.to(torch.int8)
                totals[2] += (final == expected).all(dim=1).float().mean() / len(trials)
                semantic = torch.zeros(
                    final.shape[0], dtype=torch.bool, device=self.device
                )
                for output_mode in {
                    dataset.output_mode for dataset in datasets.datasets
                }:
                    matching_tasks = torch.tensor(
                        [
                            dataset.output_mode == output_mode
                            for dataset in datasets.datasets
                        ]
                    )
                    selected = matching_tasks[task_indices].to(self.device)
                    semantic |= selected & semantic_correct(
                        final, expected, lengths, output_mode
                    )
                totals[1] += semantic.float().mean() / len(trials)
            del rollout, losses
        parameters = [
            parameter
            for group in self.optimizer.param_groups
            for parameter in group["params"]
            if parameter.grad is not None
        ]
        if self.config.training.grad_clip is None:
            gradient_norm = float(
                torch.stack(
                    [parameter.grad.detach().square().sum() for parameter in parameters]
                )
                .sum()
                .sqrt()
            )
        else:
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    parameters, self.config.training.grad_clip
                )
            )
        if not math.isfinite(gradient_norm):
            raise FloatingPointError(
                f"non-finite gradient norm at update {self.current_update + 1}"
            )
        self.optimizer.step()
        self.current_update += 1
        loss_total, semantic_total, raw_total = totals.cpu().tolist()
        return StepMetrics(
            self.current_update,
            loss_total,
            learning_rate,
            noise,
            gradient_norm,
            semantic_total,
            raw_total,
            trials,
        )

    @torch.no_grad()
    def validation_metrics(self) -> ValidationMetrics:
        was_training = self.model.training
        self.model.eval()
        losses_by_task = {name: [] for name in self.task_names}
        accuracies_by_task = {name: [] for name in self.task_names}
        devices = list(range(torch.cuda.device_count()))
        try:
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(self.config.training.seed)
                if self.device.type == "cuda":
                    torch.cuda.manual_seed_all(self.config.training.seed)
                for trial in validation_trials(
                    self.config.training,
                    self.task_names,
                    minimum_tape_slots=self.config.model.program_length,
                ):
                    datasets = self._datasets(trial)
                    layout = TapeLayout(trial.tape_slots)
                    for task_index, dataset in enumerate(datasets.datasets):
                        loss_sum = 0.0
                        correct = 0
                        for offset in range(
                            0, len(dataset), self.config.training.batch_size_per_task
                        ):
                            indices = torch.arange(
                                offset,
                                min(
                                    offset + self.config.training.batch_size_per_task,
                                    len(dataset),
                                ),
                            )
                            inputs, targets, lengths = (
                                value.to(self.device) for value in dataset.take(indices)
                            )
                            task_indices = torch.full(
                                (len(indices),),
                                task_index,
                                dtype=torch.int64,
                                device=self.device,
                            )
                            rollout = self.model(
                                self.model.initial_state(inputs, task_indices),
                                trial.steps,
                                io_only=True,
                                start_step=trial.free_steps + 1,
                            )
                            loss = supervised_loss(
                                rollout,
                                targets,
                                layout,
                                self.config.model.io_channel,
                                trial.free_steps,
                                trial.supervision_steps,
                                start_step=trial.free_steps + 1,
                            ).base
                            loss_sum += float(loss) * len(indices)
                            final = quantize(layout.extract_tape(rollout[:, -1]))
                            correct += int(
                                semantic_correct(
                                    final,
                                    targets.to(torch.int8),
                                    lengths,
                                    dataset.output_mode,
                                ).sum()
                            )
                        losses_by_task[dataset.name].append(loss_sum / len(dataset))
                        accuracies_by_task[dataset.name].append(correct / len(dataset))
        finally:
            self.model.train(was_training)
        task_losses = {
            name: sum(values) / len(values) for name, values in losses_by_task.items()
        }
        task_accuracies = {
            name: sum(values) / len(values)
            for name, values in accuracies_by_task.items()
        }
        result = sum(
            weight * task_losses[name]
            for name, weight in zip(self.task_names, self.task_weights)
        ) / sum(self.task_weights)
        if not math.isfinite(result) or not all(
            math.isfinite(value) for value in task_accuracies.values()
        ):
            raise FloatingPointError("validation produced a non-finite loss")
        return ValidationMetrics(result, task_losses, task_accuracies)

    def fit(
        self,
        checkpoint_dir: str | Path = "checkpoints",
        progress_every: int = 25,
        callback: Callable[[StepMetrics], None] | None = None,
    ) -> list[dict]:
        if progress_every < 1:
            raise ValueError("progress_every must be positive")
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        while self.current_update < self.config.training.updates:
            metrics = self.train_step()
            should_validate = (
                metrics.update % self.config.training.validation_every == 0
                or metrics.update == self.config.training.updates
            )
            validation_metrics = self.validation_metrics() if should_validate else None
            validation_loss = (
                None if validation_metrics is None else validation_metrics.loss
            )
            is_best = (
                validation_loss is not None
                and validation_loss < self.best_validation_loss
            )
            if is_best:
                self.best_validation_loss = validation_loss
            metrics = replace(
                metrics,
                validation_loss=validation_loss,
                validation_accuracies=(
                    None
                    if validation_metrics is None
                    else validation_metrics.task_accuracies
                ),
            )
            self.history.append(metrics.to_dict())
            if is_best:
                self.save(checkpoint_dir / "best.pt")
            if callback is not None:
                callback(metrics)
            if metrics.update % progress_every == 0 or should_validate:
                validation = (
                    "" if validation_loss is None else f" val={validation_loss:.6g}"
                )
                print(
                    f"update {metrics.update:5d}/{self.config.training.updates} "
                    f"loss={metrics.loss:.6g} semantic={metrics.semantic_accuracy:.2%} "
                    f"raw={metrics.raw_accuracy:.2%} lr={metrics.learning_rate:.3g}"
                    f" noise={metrics.perception_noise:.3g}{validation}"
                )
                if metrics.validation_accuracies is not None:
                    print(
                        "validation accuracy: "
                        + "  ".join(
                            f"{name}={accuracy:.2%}"
                            for name, accuracy in metrics.validation_accuracies.items()
                        )
                    )
            if (
                metrics.update % self.config.training.checkpoint_every == 0
                or metrics.update == self.config.training.updates
            ):
                self.save(checkpoint_dir / "latest.pt")
        return self.history

    def checkpoint(self) -> dict:
        state = {
            "format_version": CHECKPOINT_FORMAT,
            "config": self.config.to_dict(),
            "task_specs": self.task_specs,
            "task_names": self.task_names,
            "task_signatures": self.task_signatures,
            "task_signature_format": "indexed-v1",
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "update": self.current_update,
            "best_validation_loss": self.best_validation_loss,
            "history": self.history,
            "trial_rng_state": self.trial_generator.get_state(),
            "data_rng_state": self.data_generator.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "python_rng_state": random.getstate(),
        }
        if torch.cuda.is_available():
            state["cuda_rng_state"] = torch.cuda.get_rng_state_all()
        return state

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(self.checkpoint(), temporary)
        temporary.replace(path)

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        task_specs: tuple[tuple[str, float], ...] | None = None,
        device: str | None = None,
    ) -> "Trainer":
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        _validate_checkpoint(checkpoint, task_specs)
        config_data = checkpoint["config"]
        if device is not None:
            config_data = {
                **config_data,
                "training": {**config_data["training"], "device": device},
            }
        specs = _checkpoint_task_specs(checkpoint)
        trainer = cls(ExperimentConfig.from_dict(config_data), specs)
        trainer.model.load_state_dict(checkpoint["model"], strict=True)
        trainer.optimizer.load_state_dict(checkpoint["optimizer"])
        trainer.current_update = int(checkpoint["update"])
        trainer.best_validation_loss = float(checkpoint["best_validation_loss"])
        trainer.history = list(checkpoint["history"])
        trainer.trial_generator.set_state(checkpoint["trial_rng_state"])
        trainer.data_generator.set_state(checkpoint["data_rng_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        random.setstate(checkpoint["python_rng_state"])
        if trainer.device.type == "cuda" and "cuda_rng_state" in checkpoint:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
        return trainer


def _validate_checkpoint(
    checkpoint: dict,
    task_specs: tuple[tuple[str, float], ...] | None,
) -> None:
    if checkpoint.get("format_version") != CHECKPOINT_FORMAT:
        raise ValueError("checkpoint uses an incompatible model representation")
    stored = tuple(checkpoint.get("task_names", ()))
    if not stored:
        raise ValueError("checkpoint does not define its ordered tasks")
    stored_specs = _checkpoint_task_specs(checkpoint)
    stored_names, _ = validate_task_specs(stored_specs)
    if stored_names != stored:
        raise ValueError("checkpoint task specs and task order differ")
    if task_specs is not None:
        expected_names, expected_weights = validate_task_specs(task_specs)
        if expected_names != stored or tuple(expected_weights) != tuple(
            weight for _, weight in stored_specs
        ):
            raise ValueError("checkpoint task specs do not match")
    config = ExperimentConfig.from_dict(checkpoint["config"])
    signature_format = checkpoint.get("task_signature_format")
    if signature_format not in {None, "indexed-v1"}:
        raise ValueError("checkpoint uses an unknown task signature format")
    if tuple(checkpoint.get("task_signatures", ())) != _task_signatures(
        config, stored, legacy=signature_format is None
    ):
        raise ValueError("checkpoint task definitions do not match")


def _checkpoint_task_specs(checkpoint: dict) -> tuple[tuple[str, float], ...]:
    return tuple(tuple(spec) for spec in checkpoint.get("task_specs", ()))


def load_model(
    path: str | Path,
    device: str = "auto",
) -> tuple[NeuralCellularAutomaton, ExperimentConfig, dict]:
    resolved = resolve_device(device)
    checkpoint = torch.load(path, map_location=resolved, weights_only=False)
    _validate_checkpoint(checkpoint, None)
    config = ExperimentConfig.from_dict(checkpoint["config"])
    model = NeuralCellularAutomaton(config.model, tuple(checkpoint["task_names"])).to(
        resolved
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, config, checkpoint


def train_seeds(
    config: ExperimentConfig,
    task_specs: tuple[tuple[str, float], ...],
    seeds: tuple[int, ...],
    checkpoint_dir: str | Path = "checkpoints",
    *,
    resume: bool = False,
    progress_every: int = 25,
) -> list[SeedResult]:
    task_names, _ = validate_task_specs(task_specs)
    if not seeds or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("seeds must contain non-negative integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique")
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in seeds:
        model_config = config.model
        if model_config.learnable_kernel_init == "random":
            model_config = replace(model_config, random_kernel_seed=seed)
        seed_config = replace(
            config,
            model=model_config,
            training=replace(config.training, seed=seed),
        )
        seed_dir = checkpoint_dir / f"seed_{seed}"
        latest = seed_dir / "latest.pt"
        if resume and latest.is_file():
            trainer = Trainer.from_checkpoint(
                latest, task_specs, seed_config.training.device
            )
            if trainer.config != seed_config:
                raise ValueError(f"seed {seed} checkpoint configuration does not match")
        else:
            trainer = Trainer(seed_config, task_specs)
        history = trainer.fit(seed_dir, progress_every=progress_every)
        results.append(
            SeedResult(
                seed, seed_dir / "best.pt", trainer.best_validation_loss, history
            )
        )
    best = min(results, key=lambda result: result.best_validation_loss)
    shutil.copy2(best.checkpoint, checkpoint_dir / "best.pt")
    return results
