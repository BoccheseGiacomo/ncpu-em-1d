from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


TERNARY_THRESHOLD = 0.333
SYMBOL_TO_VALUE = {"0": -1.0, "B": 0.0, "1": 1.0}
VALUE_TO_SYMBOL = {-1: "0", 0: "B", 1: "1"}


def validate_symbols(symbols: str, *, allow_empty: bool = False) -> None:
    if not isinstance(symbols, str):
        raise TypeError("symbols must be a string")
    if not symbols and not allow_empty:
        raise ValueError("symbol strings cannot be empty")
    unknown = set(symbols) - set(SYMBOL_TO_VALUE)
    if unknown:
        raise ValueError(f"unsupported symbols: {sorted(unknown)}")


def encode_strings(strings: Sequence[str], tape_slots: int) -> torch.Tensor:
    if type(tape_slots) is not int or tape_slots < 1:
        raise ValueError("tape_slots must be a positive integer")
    if not strings:
        raise ValueError("at least one string is required")
    encoded = torch.zeros(len(strings), tape_slots, dtype=torch.float32)
    for row, symbols in enumerate(strings):
        validate_symbols(symbols, allow_empty=True)
        if len(symbols) > tape_slots:
            raise ValueError(
                f"string of length {len(symbols)} exceeds {tape_slots} tape slots"
            )
        if symbols:
            encoded[row, : len(symbols)] = torch.tensor(
                [SYMBOL_TO_VALUE[symbol] for symbol in symbols]
            )
    return encoded


def quantize(values: torch.Tensor) -> torch.Tensor:
    values = torch.as_tensor(values)
    if not torch.is_floating_point(values):
        raise ValueError("values must be floating point")
    if not torch.isfinite(values).all():
        raise ValueError("values must be finite")
    return (values > TERNARY_THRESHOLD).to(torch.int8) - (
        values < -TERNARY_THRESHOLD
    ).to(torch.int8)


def tensor_to_symbols(values: torch.Tensor) -> str:
    values = torch.as_tensor(values)
    if values.ndim != 1:
        raise ValueError("single-example inference expects a one-dimensional tape")
    return "".join(VALUE_TO_SYMBOL[value] for value in quantize(values).cpu().tolist())


def integer_to_binary(value: int) -> str:
    if type(value) is not int or value < 0:
        raise ValueError("value must be a non-negative integer")
    return format(value, "b")


def binary_to_integer(symbols: str) -> int:
    if not symbols or set(symbols) - {"0", "1"}:
        raise ValueError("a binary string must contain one or more 0/1 symbols")
    if len(symbols) > 1 and symbols.startswith("0"):
        raise ValueError("integer strings must use minimal binary representation")
    return int(symbols, 2)


@dataclass(frozen=True)
class InterpretedTape:
    raw: str
    binary_strings: tuple[str, ...]
    integers: tuple[int, ...]
    terminated: bool
    valid: bool
    integer_valid: bool


def interpret_tape(values: torch.Tensor, mode: str = "single") -> InterpretedTape:
    if mode not in {"single", "multiple"}:
        raise ValueError("mode must be 'single' or 'multiple'")
    raw = tensor_to_symbols(values)
    if mode == "single":
        end = raw.find("B")
        terminated = end >= 0
        binary_strings = (raw[:end] if terminated else raw,)
        valid = True
    else:
        end = raw.find("BB")
        terminated = end >= 0
        prefix = raw[:end] if terminated else raw
        binary_strings = tuple(prefix.split("B")) if prefix else ("",)
        valid = all(part and set(part) <= {"0", "1"} for part in binary_strings)
    integer_valid = valid and all(
        value and (len(value) == 1 or value.startswith("1")) for value in binary_strings
    )
    integers = (
        tuple(binary_to_integer(value) for value in binary_strings)
        if integer_valid
        else ()
    )
    return InterpretedTape(
        raw, binary_strings, integers, terminated, valid, integer_valid
    )


@dataclass(frozen=True)
class TapeLayout:
    tape_slots: int

    def __post_init__(self) -> None:
        if type(self.tape_slots) is not int or self.tape_slots < 1:
            raise ValueError("tape_slots must be a positive integer")

    def extract_tape(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim < 1 or values.shape[-1] != self.tape_slots:
            raise ValueError(f"tape must end with a dimension of {self.tape_slots}")
        return values

    def schema(self) -> str:
        return (
            " ".join("T" for _ in range(self.tape_slots))
            + "\nT: shared mutable I/O cell"
        )
