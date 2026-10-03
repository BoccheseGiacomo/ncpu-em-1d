from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import TestCase
from .model import NeuralCellularAutomaton
from .tape import InterpretedTape, TapeLayout, encode_strings, interpret_tape, quantize
from .tasks import MultiTaskDataset, TaskDataset, binary_tasks, semantic_correct


@dataclass(frozen=True)
class EvaluationResult:
    case: str
    task: str
    examples: int
    total_examples: int
    tape_slots: int
    batch_size: int
    seed: int
    step_start: int
    step_end: int
    mean_mse: float
    mean_semantic_accuracy: float
    mean_raw_accuracy: float
    mean_symbol_accuracy: float
    stable_semantic_accuracy: float
    stable_raw_accuracy: float
    best_semantic_step: int
    best_semantic_accuracy: float
    best_mse_step: int
    best_mse: float
    mse_by_step: tuple[float, ...]
    semantic_by_step: tuple[float, ...]
    raw_by_step: tuple[float, ...]
    symbol_by_step: tuple[float, ...]

    def summary(self) -> dict[str, float | int | str]:
        return {
            "case": self.case,
            "task": self.task,
            "examples": self.examples,
            "total_examples": self.total_examples,
            "tape_slots": self.tape_slots,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "steps": f"{self.step_start}-{self.step_end}",
            "mse": self.mean_mse,
            "semantic": self.mean_semantic_accuracy,
            "raw": self.mean_raw_accuracy,
            "symbol": self.mean_symbol_accuracy,
            "stable_semantic": self.stable_semantic_accuracy,
            "stable_raw": self.stable_raw_accuracy,
        }


def _sample_indices(population: int, count: int, seed: int) -> torch.Tensor:
    # Floyd's algorithm: uniform sampling without replacement, O(count) memory.
    generator = torch.Generator().manual_seed(seed)
    selected: set[int] = set()
    ordered = []
    for upper in range(population - count, population):
        draw = int(torch.randint(upper + 1, (), generator=generator))
        index = upper if draw in selected else draw
        selected.add(index)
        ordered.append(index)
    indices = torch.tensor(ordered, dtype=torch.int64)
    return indices[torch.randperm(count, generator=generator)]


@torch.no_grad()
def evaluate(
    model: NeuralCellularAutomaton,
    dataset: TaskDataset,
    *,
    steps: int,
    step_start: int,
    step_end: int,
    batch_size: int = 256,
    seed: int = 0,
    max_examples: int | None = None,
    case_name: str = "custom",
) -> EvaluationResult:
    task_index = model.task_index(dataset.name)
    if dataset.tape_slots < model.config.program_length:
        raise ValueError("evaluation tape cannot contain the complete program")
    if any(type(value) is not int for value in (steps, step_start, step_end)):
        raise TypeError("evaluation steps must be integers")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be positive")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not 0 <= step_start <= step_end <= steps:
        raise ValueError("evaluation steps must satisfy 0 <= start <= end <= steps")
    if max_examples is not None and (type(max_examples) is not int or max_examples < 1):
        raise ValueError("max_examples must be a positive integer or None")
    if not isinstance(case_name, str) or not case_name.strip():
        raise ValueError("case_name must be a non-empty string")

    count = len(dataset) if max_examples is None else min(max_examples, len(dataset))
    indices = (
        _sample_indices(len(dataset), count, seed) if count < len(dataset) else None
    )
    step_count = steps + 1
    mse_sums = torch.zeros(step_count, dtype=torch.float64)
    semantic_sums = torch.zeros(step_count, dtype=torch.float64)
    raw_sums = torch.zeros(step_count, dtype=torch.float64)
    symbol_sums = torch.zeros(step_count, dtype=torch.float64)
    stable_semantic_sum = stable_raw_sum = 0.0
    layout = TapeLayout(dataset.tape_slots)
    device = model.device
    was_training = model.training
    model.eval()
    devices = list(range(torch.cuda.device_count()))
    window = slice(step_start, step_end + 1)
    try:
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            for offset in range(0, count, batch_size):
                batch_indices = (
                    torch.arange(offset, min(offset + batch_size, count))
                    if indices is None
                    else indices[offset : offset + batch_size]
                )
                inputs, targets, lengths = (
                    value.to(device) for value in dataset.take(batch_indices)
                )
                task_indices = torch.full(
                    (len(batch_indices),), task_index, dtype=torch.int64, device=device
                )
                rollout = model(
                    model.initial_state(inputs, task_indices), steps, io_only=True
                )
                values = layout.extract_tape(rollout)
                expected = targets.unsqueeze(1)
                discrete = quantize(values)
                discrete_targets = targets.to(torch.int8)
                correct_symbols = discrete == discrete_targets.unsqueeze(1)
                raw = correct_symbols.all(dim=-1)
                semantic = semantic_correct(
                    discrete, discrete_targets, lengths, dataset.output_mode
                )
                mse_sums += (values - expected).square().sum(dim=(0, 2)).cpu()
                symbol_sums += correct_symbols.sum(dim=(0, 2)).cpu()
                raw_sums += raw.sum(dim=0).cpu()
                semantic_sums += semantic.sum(dim=0).cpu()
                stable_semantic_sum += float(semantic[:, window].all(dim=1).sum())
                stable_raw_sum += float(raw[:, window].all(dim=1).sum())
    finally:
        model.train(was_training)

    mse_curve = mse_sums / (count * dataset.tape_slots)
    semantic_curve = semantic_sums / count
    raw_curve = raw_sums / count
    symbol_curve = symbol_sums / (count * dataset.tape_slots)
    mse_window = mse_curve[window]
    semantic_window = semantic_curve[window]
    best_semantic_offset = int(semantic_window.argmax())
    best_mse_offset = int(mse_window.argmin())
    return EvaluationResult(
        case_name,
        dataset.name,
        count,
        len(dataset),
        dataset.tape_slots,
        batch_size,
        seed,
        step_start,
        step_end,
        float(mse_window.mean()),
        float(semantic_window.mean()),
        float(raw_curve[window].mean()),
        float(symbol_curve[window].mean()),
        stable_semantic_sum / count,
        stable_raw_sum / count,
        step_start + best_semantic_offset,
        float(semantic_window[best_semantic_offset]),
        step_start + best_mse_offset,
        float(mse_window[best_mse_offset]),
        tuple(map(float, mse_curve)),
        tuple(map(float, semantic_curve)),
        tuple(map(float, raw_curve)),
        tuple(map(float, symbol_curve)),
    )


@dataclass(frozen=True)
class InferenceResult:
    task: str
    input: str
    steps: int
    values: torch.Tensor
    interpreted: InterpretedTape


@torch.no_grad()
def infer(
    model: NeuralCellularAutomaton,
    input_symbols: str,
    *,
    tape_slots: int,
    task_name: str,
    steps: int,
    output_mode: str = "single",
) -> InferenceResult:
    if type(steps) is not int or steps < 0:
        raise ValueError("steps must be a non-negative integer")
    if output_mode not in {"single", "multiple"}:
        raise ValueError("output_mode must be 'single' or 'multiple'")
    layout = TapeLayout(tape_slots)
    encoded = encode_strings((input_symbols,), tape_slots).to(model.device)
    task_indices = torch.tensor([model.task_index(task_name)], device=model.device)
    was_training = model.training
    model.eval()
    try:
        final = model(model.initial_state(encoded, task_indices), steps)[
            :, -1, model.config.io_channel
        ]
        values = layout.extract_tape(final)[0].cpu()
    finally:
        model.train(was_training)
    return InferenceResult(
        task_name, input_symbols, steps, values, interpret_tape(values, output_mode)
    )


@torch.no_grad()
def evaluate_tasks(
    model: NeuralCellularAutomaton,
    datasets: MultiTaskDataset,
    **kwargs,
) -> list[EvaluationResult]:
    if datasets.task_names != model.task_names:
        raise ValueError("evaluation task order differs from the model")
    results = [evaluate(model, dataset, **kwargs) for dataset in datasets.datasets]
    curves = ("mse_by_step", "semantic_by_step", "raw_by_step", "symbol_by_step")
    averaged = {
        name: tuple(
            sum(values) / len(values)
            for values in zip(*(getattr(result, name) for result in results))
        )
        for name in curves
    }
    window = slice(results[0].step_start, results[0].step_end + 1)
    mse_window = averaged["mse_by_step"][window]
    semantic_window = averaged["semantic_by_step"][window]
    best_mse_offset = min(range(len(mse_window)), key=mse_window.__getitem__)
    best_semantic_offset = max(
        range(len(semantic_window)), key=semantic_window.__getitem__
    )
    aggregate = EvaluationResult(
        case=results[0].case,
        task="aggregate",
        examples=sum(result.examples for result in results),
        total_examples=sum(result.total_examples for result in results),
        tape_slots=datasets.tape_slots,
        batch_size=results[0].batch_size,
        seed=results[0].seed,
        step_start=results[0].step_start,
        step_end=results[0].step_end,
        mean_mse=sum(result.mean_mse for result in results) / len(results),
        mean_semantic_accuracy=sum(result.mean_semantic_accuracy for result in results)
        / len(results),
        mean_raw_accuracy=sum(result.mean_raw_accuracy for result in results)
        / len(results),
        mean_symbol_accuracy=sum(result.mean_symbol_accuracy for result in results)
        / len(results),
        stable_semantic_accuracy=sum(
            result.stable_semantic_accuracy for result in results
        )
        / len(results),
        stable_raw_accuracy=sum(result.stable_raw_accuracy for result in results)
        / len(results),
        best_semantic_step=results[0].step_start + best_semantic_offset,
        best_semantic_accuracy=semantic_window[best_semantic_offset],
        best_mse_step=results[0].step_start + best_mse_offset,
        best_mse=mse_window[best_mse_offset],
        **averaged,
    )
    return [*results, aggregate]


def evaluate_cases(
    model: NeuralCellularAutomaton,
    task_names: tuple[str, ...],
    cases: tuple[TestCase, ...],
    *,
    batch_size: int = 256,
    seed: int = 0,
    max_examples: int | None = None,
) -> list[EvaluationResult]:
    results = []
    for case in cases:
        case.validate()
        if case.tape_slots < model.config.program_length:
            raise ValueError(f"test case {case.name!r} cannot contain the program")
        try:
            datasets = MultiTaskDataset.from_tasks(
                binary_tasks(task_names, case.input_length, include_shorter=False),
                case.tape_slots,
            )
        except ValueError as error:
            raise ValueError(f"test case {case.name!r}: {error}") from error
        results.extend(
            evaluate_tasks(
                model,
                datasets,
                steps=case.steps,
                step_start=case.free_steps + 1,
                step_end=case.steps,
                batch_size=batch_size,
                seed=seed,
                max_examples=max_examples,
                case_name=case.name,
            )
        )
    return results


def format_results(results: list[EvaluationResult]) -> str:
    lines = [
        "case              task             n  tape   steps      mse    "
        "semantic      raw   symbol   stable"
    ]
    for result in results:
        lines.append(
            f"{result.case[:17]:<17} {result.task[:16]:<16} {result.examples:>5} "
            f"{result.tape_slots:>5}  {result.step_start:>3}-{result.step_end:<3} "
            f"{result.mean_mse:>9.6f}  {result.mean_semantic_accuracy:>8.2%} "
            f"{result.mean_raw_accuracy:>8.2%} {result.mean_symbol_accuracy:>8.2%} "
            f"{result.stable_semantic_accuracy:>8.2%}"
        )
    return "\n".join(lines)
