from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from .tape import encode_strings, integer_to_binary, validate_symbols


@dataclass(frozen=True)
class StringExample:
    input: str
    target: str


def _index(index: int, count: int) -> int:
    if not isinstance(index, int):
        raise TypeError("example index must be an integer")
    index = index + count if index < 0 else index
    if not 0 <= index < count:
        raise IndexError("example index out of range")
    return index


@dataclass(frozen=True)
class BinaryStrings(Sequence[str]):
    max_length: int
    include_shorter: bool = True
    include_empty: bool = True

    def __post_init__(self) -> None:
        if type(self.max_length) is not int or not 0 <= self.max_length <= 62:
            raise ValueError("indexed binary lengths must be integers in [0, 62]")
        if (
            type(self.include_shorter) is not bool
            or type(self.include_empty) is not bool
        ):
            raise TypeError("binary domain flags must be booleans")
        if self.max_length and self.include_empty and not self.include_shorter:
            raise ValueError("include_empty requires include_shorter")

    def __len__(self) -> int:
        if self.max_length == 0:
            return 1
        if self.include_shorter:
            return (1 << (self.max_length + 1)) - 2 + int(self.include_empty)
        return 1 << self.max_length

    def __getitem__(self, index):
        if isinstance(index, slice):
            return tuple(self[i] for i in range(*index.indices(len(self))))
        index = _index(index, len(self))
        if self.max_length == 0:
            return ""
        if not self.include_shorter:
            return format(index, f"0{self.max_length}b")
        rank = index + 1 + int(not self.include_empty)
        length = rank.bit_length() - 1
        return format(rank - (1 << length), f"0{length}b") if length else ""


def _bit_not(value: str) -> str:
    return value.translate(str.maketrans("01", "10"))


def _gray_encode(value: str) -> str:
    if not value:
        return ""
    return value[0] + "".join(
        "1" if left != right else "0" for left, right in zip(value, value[1:])
    )


def _prefix_xor(value: str) -> str:
    parity = 0
    output = []
    for bit in value:
        parity ^= int(bit)
        output.append(str(parity))
    return "".join(output)


def _increment(value: str) -> str:
    bits = list(value)
    for index in range(len(bits) - 1, -1, -1):
        if bits[index] == "0":
            bits[index] = "1"
            break
        bits[index] = "0"
    return "".join(bits)


TASK_TRANSFORMS: dict[str, Callable[[str], str]] = {
    "copy": lambda value: value,
    "bit_not": _bit_not,
    "reverse": lambda value: value[::-1],
    "reverse_not": lambda value: _bit_not(value[::-1]),
    "shift_left_zero": lambda value: value[1:] + "0" if value else "",
    "shift_right_zero": lambda value: "0" + value[:-1] if value else "",
    "gray_encode": _gray_encode,
    "prefix_xor": _prefix_xor,
    "increment": _increment,
    "parity": lambda value: "1" if value.count("1") % 2 else "0",
    "append_0": lambda value: value + "0",
    "append_1": lambda value: value + "1",
}
# Increment when built-in target semantics change.
TASK_DEFINITION_VERSION = 1


def task_target(name: str, value: str) -> str:
    if name not in TASK_TRANSFORMS:
        raise ValueError(f"unknown task: {name!r}")
    validate_symbols(value, allow_empty=True)
    if "B" in value:
        raise ValueError("these binary tasks require 0/1 inputs")
    return TASK_TRANSFORMS[name](value)


def output_length_bound(name: str, input_max_length: int) -> int:
    if name not in TASK_TRANSFORMS:
        raise ValueError(f"unknown task: {name!r}")
    if type(input_max_length) is not int or input_max_length < 0:
        raise ValueError("input maximum must be a non-negative integer")
    if name == "parity":
        return 1
    if name in {"append_0", "append_1"}:
        return input_max_length + 1
    return input_max_length


@dataclass(frozen=True)
class BinaryExamples(Sequence[StringExample]):
    task_name: str
    strings: BinaryStrings

    def __post_init__(self) -> None:
        if self.task_name not in TASK_TRANSFORMS:
            raise ValueError(f"unknown task: {self.task_name!r}")

    def __len__(self) -> int:
        return len(self.strings)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return tuple(self[i] for i in range(*index.indices(len(self))))
        value = self.strings[index]
        return StringExample(value, task_target(self.task_name, value))

    @property
    def maximum_lengths(self) -> tuple[int, int]:
        maximum = self.strings.max_length
        return maximum, output_length_bound(self.task_name, maximum)


@dataclass(frozen=True)
class AdditionExamples(Sequence[StringExample]):
    operand_bits: int

    def __post_init__(self) -> None:
        if type(self.operand_bits) is not int or not 1 <= self.operand_bits <= 31:
            raise ValueError("indexed addition operand bits must be in [1, 31]")

    def __len__(self) -> int:
        return 1 << (2 * self.operand_bits)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return tuple(self[i] for i in range(*index.indices(len(self))))
        a, b = divmod(_index(index, len(self)), 1 << self.operand_bits)
        return StringExample(
            f"{integer_to_binary(a)}B{integer_to_binary(b)}", integer_to_binary(a + b)
        )

    @property
    def maximum_lengths(self) -> tuple[int, int]:
        return 2 * self.operand_bits + 1, self.operand_bits + 1


@dataclass(frozen=True)
class ExampleStrings(Sequence[str]):
    examples: Sequence[StringExample]
    field: str

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return tuple(self[i] for i in range(*index.indices(len(self))))
        return getattr(self.examples[index], self.field)


@dataclass(frozen=True)
class StringTask:
    name: str
    examples: Sequence[StringExample]
    output_mode: str = "single"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("task name cannot be empty")
        if not self.examples:
            raise ValueError("a task must contain at least one example")
        if self.output_mode not in {"single", "multiple"}:
            raise ValueError("output_mode must be 'single' or 'multiple'")
        if isinstance(self.examples, (BinaryExamples, AdditionExamples)):
            expected = (
                self.examples.task_name
                if isinstance(self.examples, BinaryExamples)
                else f"addition-{self.examples.operand_bits}-bit"
            )
            if self.name != expected or self.output_mode != "single":
                raise ValueError(
                    "generated task metadata does not match its definition"
                )
            return
        for example in self.examples:
            validate_symbols(example.input, allow_empty=True)
            validate_symbols(example.target, allow_empty=True)
            if example.target and (
                example.target.startswith("B") or example.target.endswith("B")
            ):
                raise ValueError("targets cannot begin or end with B")
            if self.output_mode == "single" and "B" in example.target:
                raise ValueError("single-output targets cannot contain B")
            if self.output_mode == "multiple" and example.target:
                if "BB" in example.target:
                    raise ValueError("multiple-output values use single-B separators")
                if any(not part for part in example.target.split("B")):
                    raise ValueError("multiple-output targets contain an empty value")

    @property
    def maximum_lengths(self) -> tuple[int, int]:
        if isinstance(self.examples, (BinaryExamples, AdditionExamples)):
            return self.examples.maximum_lengths
        return (
            max(len(example.input) for example in self.examples),
            max(len(example.target) for example in self.examples),
        )


def addition_task(operand_bits: int) -> StringTask:
    return StringTask(f"addition-{operand_bits}-bit", AdditionExamples(operand_bits))


def binary_strings(
    max_length: int, *, include_shorter: bool = True, include_empty: bool = True
) -> BinaryStrings:
    return BinaryStrings(max_length, include_shorter, include_empty)


def _binary_task(name: str, max_length: int, include_shorter: bool) -> StringTask:
    return StringTask(
        name,
        BinaryExamples(
            name, BinaryStrings(max_length, include_shorter, include_shorter)
        ),
    )


def reverse_task(max_length: int, *, include_shorter: bool = True) -> StringTask:
    return _binary_task("reverse", max_length, include_shorter)


def bitwise_not_task(max_length: int, *, include_shorter: bool = True) -> StringTask:
    return _binary_task("bit_not", max_length, include_shorter)


def parity_task(max_length: int, *, include_shorter: bool = True) -> StringTask:
    return _binary_task("parity", max_length, include_shorter)


def copy_task(max_length: int, *, include_shorter: bool = True) -> StringTask:
    return _binary_task("copy", max_length, include_shorter)


def reverse_not_task(max_length: int, *, include_shorter: bool = True) -> StringTask:
    return _binary_task("reverse_not", max_length, include_shorter)


def shift_left_zero_task(
    max_length: int, *, include_shorter: bool = True
) -> StringTask:
    return _binary_task("shift_left_zero", max_length, include_shorter)


def shift_right_zero_task(
    max_length: int, *, include_shorter: bool = True
) -> StringTask:
    return _binary_task("shift_right_zero", max_length, include_shorter)


def gray_encode_task(max_length: int, *, include_shorter: bool = True) -> StringTask:
    return _binary_task("gray_encode", max_length, include_shorter)


def prefix_xor_task(max_length: int, *, include_shorter: bool = True) -> StringTask:
    return _binary_task("prefix_xor", max_length, include_shorter)


def increment_task(max_length: int, *, include_shorter: bool = True) -> StringTask:
    return _binary_task("increment", max_length, include_shorter)


def append_task(
    max_length: int, bit: str, *, include_shorter: bool = True
) -> StringTask:
    if bit not in {"0", "1"}:
        raise ValueError("appended bit must be '0' or '1'")
    return _binary_task(f"append_{bit}", max_length, include_shorter)


def append_zero_task(max_length: int, *, include_shorter: bool = True) -> StringTask:
    return append_task(max_length, "0", include_shorter=include_shorter)


def append_one_task(max_length: int, *, include_shorter: bool = True) -> StringTask:
    return append_task(max_length, "1", include_shorter=include_shorter)


TASK_FACTORIES = {
    "copy": copy_task,
    "bit_not": bitwise_not_task,
    "reverse": reverse_task,
    "reverse_not": reverse_not_task,
    "shift_left_zero": shift_left_zero_task,
    "shift_right_zero": shift_right_zero_task,
    "gray_encode": gray_encode_task,
    "prefix_xor": prefix_xor_task,
    "increment": increment_task,
    "parity": parity_task,
    "append_0": append_zero_task,
    "append_1": append_one_task,
}


def binary_tasks(
    names: tuple[str, ...], max_length: int, *, include_shorter: bool = True
) -> tuple[StringTask, ...]:
    if not names or any(not isinstance(name, str) for name in names):
        raise ValueError("task names must be non-empty strings")
    if len(set(names)) != len(names):
        raise ValueError("task names must be non-empty and unique")
    unknown = set(names) - set(TASK_FACTORIES)
    if unknown:
        raise ValueError(f"unknown tasks: {sorted(unknown)}")
    return tuple(
        TASK_FACTORIES[name](max_length, include_shorter=include_shorter)
        for name in names
    )


@dataclass(frozen=True)
class TaskDataset:
    task: StringTask
    tape_slots: int

    def __post_init__(self) -> None:
        if type(self.tape_slots) is not int or self.tape_slots < 1:
            raise ValueError("tape_slots must be positive")
        input_max, output_max = self.task.maximum_lengths
        if input_max > self.tape_slots - 1:
            raise ValueError(
                f"task {self.name!r} has an input that does not leave "
                "one blank tape cell"
            )
        if output_max > self.tape_slots - 1:
            raise ValueError(
                f"task {self.name!r} has an output that does not leave "
                "one blank tape cell"
            )

    @classmethod
    def from_task(cls, task: StringTask, tape_slots: int) -> "TaskDataset":
        return cls(task, tape_slots)

    @property
    def name(self) -> str:
        return self.task.name

    @property
    def output_mode(self) -> str:
        return self.task.output_mode

    @property
    def input_strings(self) -> Sequence[str]:
        if isinstance(self.task.examples, BinaryExamples):
            return self.task.examples.strings
        return ExampleStrings(self.task.examples, "input")

    @property
    def target_strings(self) -> Sequence[str]:
        return ExampleStrings(self.task.examples, "target")

    def __len__(self) -> int:
        return len(self.task.examples)

    @property
    def signature(self) -> str:
        examples = self.task.examples
        if isinstance(examples, BinaryExamples):
            domain = examples.strings
            data = {
                "format": "indexed-binary-v1",
                "name": self.name,
                "output_mode": self.output_mode,
                "definition": TASK_DEFINITION_VERSION,
                "max_length": domain.max_length,
                "include_shorter": domain.include_shorter,
                "include_empty": domain.include_empty,
            }
        elif isinstance(examples, AdditionExamples):
            data = {
                "format": "indexed-addition-v1",
                "name": self.name,
                "definition": TASK_DEFINITION_VERSION,
                "operand_bits": examples.operand_bits,
            }
        else:
            return self.legacy_signature()
        return hashlib.sha256(
            json.dumps(data, separators=(",", ":"), ensure_ascii=True).encode()
        ).hexdigest()

    def legacy_signature(self) -> str:
        # Verify old checkpoints without building tuples, JSON lists, or tensors.
        digest = hashlib.sha256()
        prefix = (
            '{"name":'
            + json.dumps(self.name, ensure_ascii=True)
            + ',"output_mode":'
            + json.dumps(self.output_mode, ensure_ascii=True)
        )
        digest.update(prefix.encode())
        for field in ("input", "target"):
            digest.update((',"' + field + 's":[').encode())
            for index, example in enumerate(self.task.examples):
                if index:
                    digest.update(b",")
                digest.update(
                    json.dumps(getattr(example, field), ensure_ascii=True).encode()
                )
            digest.update(b"]")
        digest.update(b"}")
        return digest.hexdigest()

    def take(self, indices: torch.Tensor) -> tuple[torch.Tensor, ...]:
        indices = torch.as_tensor(indices, dtype=torch.int64, device="cpu")
        if indices.ndim != 1:
            raise ValueError("example indices must be one-dimensional")
        examples = [self.task.examples[index] for index in indices.tolist()]
        if not examples:
            return (
                torch.empty(0, self.tape_slots, dtype=torch.float32),
                torch.empty(0, self.tape_slots, dtype=torch.float32),
                torch.empty(0, dtype=torch.int64),
            )
        return (
            encode_strings(
                tuple(example.input for example in examples), self.tape_slots
            ),
            encode_strings(
                tuple(example.target for example in examples), self.tape_slots
            ),
            torch.tensor(
                [len(example.target) for example in examples], dtype=torch.int64
            ),
        )

    def sample(
        self, batch_size: int, generator: torch.Generator
    ) -> tuple[torch.Tensor, ...]:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be positive")
        indices = torch.randint(len(self), (batch_size,), generator=generator)
        return self.take(indices)


@dataclass(frozen=True)
class MultiTaskDataset:
    datasets: tuple[TaskDataset, ...]

    @classmethod
    def from_tasks(
        cls, tasks: tuple[StringTask, ...], tape_slots: int
    ) -> "MultiTaskDataset":
        return cls(tuple(TaskDataset.from_task(task, tape_slots) for task in tasks))

    def __post_init__(self) -> None:
        if not self.datasets:
            raise ValueError("at least one task dataset is required")
        if len(set(self.task_names)) != len(self.datasets):
            raise ValueError("task dataset names must be unique")
        first = self.datasets[0]
        for dataset in self.datasets[1:]:
            if dataset.tape_slots != first.tape_slots:
                raise ValueError("all tasks must use the same tape capacity")
            left, right = dataset.input_strings, first.input_strings
            same_inputs = (
                left == right
                if isinstance(left, BinaryStrings) and isinstance(right, BinaryStrings)
                else len(left) == len(right)
                and all(a == b for a, b in zip(left, right))
            )
            if not same_inputs:
                raise ValueError("all tasks must use the same ordered inputs")

    @property
    def task_names(self) -> tuple[str, ...]:
        return tuple(dataset.name for dataset in self.datasets)

    @property
    def tape_slots(self) -> int:
        return self.datasets[0].tape_slots

    @property
    def signatures(self) -> tuple[str, ...]:
        return tuple(dataset.signature for dataset in self.datasets)

    def __len__(self) -> int:
        return sum(map(len, self.datasets))

    def __getitem__(self, task_name: str) -> TaskDataset:
        try:
            return self.datasets[self.task_names.index(task_name)]
        except ValueError as error:
            raise KeyError(task_name) from error

    def balanced_sample(
        self, batch_size_per_task: int, generator: torch.Generator
    ) -> tuple[torch.Tensor, ...]:
        if type(batch_size_per_task) is not int or batch_size_per_task < 1:
            raise ValueError("batch_size_per_task must be a positive integer")
        batches = []
        for task_index, dataset in enumerate(self.datasets):
            inputs, targets, lengths = dataset.sample(batch_size_per_task, generator)
            task_indices = torch.full(
                (batch_size_per_task,), task_index, dtype=torch.int64
            )
            batches.append((inputs, targets, lengths, task_indices))
        combined = tuple(torch.cat(parts) for parts in zip(*batches))
        permutation = torch.randperm(combined[0].shape[0], generator=generator)
        return tuple(value[permutation] for value in combined)


def semantic_correct(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_lengths: torch.Tensor,
    output_mode: str,
) -> torch.Tensor:
    prediction = torch.as_tensor(prediction)
    target = torch.as_tensor(target, device=prediction.device)
    target_lengths = torch.as_tensor(
        target_lengths, dtype=torch.int64, device=prediction.device
    )
    squeeze_time = prediction.ndim == 2
    if squeeze_time:
        prediction = prediction.unsqueeze(1)
    if prediction.ndim != 3:
        raise ValueError(
            "prediction must have shape (batch, time, tape) or (batch, tape)"
        )
    batch, _, tape_slots = prediction.shape
    if target.shape != (batch, tape_slots):
        raise ValueError("target shape does not match prediction")
    if target_lengths.shape != (batch,):
        raise ValueError("target_lengths shape does not match prediction")
    if output_mode not in {"single", "multiple"}:
        raise ValueError("output_mode must be 'single' or 'multiple'")
    terminator_width = 1 if output_mode == "single" else 2
    required_end = torch.minimum(
        target_lengths + terminator_width,
        torch.full_like(target_lengths, tape_slots),
    )
    positions = torch.arange(tape_slots, device=prediction.device).view(1, 1, -1)
    required = positions < required_end.view(batch, 1, 1)
    correct = ((prediction == target.unsqueeze(1)) | ~required).all(dim=-1)
    return correct.squeeze(1) if squeeze_time else correct
