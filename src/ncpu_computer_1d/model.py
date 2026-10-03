from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig


def perception_kernel(name: str, radius: int = 1, random_seed: int = 0) -> torch.Tensor:
    if radius not in {1, 2}:
        raise ValueError("radius must be 1 or 2")
    if name == "random":
        generator = torch.Generator().manual_seed(random_seed)
        kernel = torch.randn(2 * radius + 1, generator=generator)
        return kernel / kernel.norm()
    if radius == 1:
        bank = {
            "identity": [0, 1, 0],
            "sobel_x": [-0.5, 0, 0.5],
            "laplacian": [0.5, -1, 0.5],
            "neighbor_mean": [0.5, 0, 0.5],
        }
    else:
        bank = {
            "identity": [0, 0, 1, 0, 0],
            "sobel_x": [-0.25, -0.5, 0, 0.5, 0.25],
            "outer_derivative": [-0.5, 0, 0, 0, 0.5],
            "neighbor_mean": [0.25, 0.25, 0, 0.25, 0.25],
            "laplacian": [0.25, 0.25, -1, 0.25, 0.25],
            "neighbor_contrast": [-0.5, 0.5, 0, 0.5, -0.5],
        }
    if name not in bank:
        raise ValueError(f"unknown radius-{radius} perception kernel: {name}")
    return torch.tensor(bank[name], dtype=torch.float32)


class Perception(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.channels = config.channels
        self.radius = config.radius
        self.fixed_names = config.fixed_kernels
        self.learnable_count = config.learned_kernel_count
        for index, name in enumerate(self.fixed_names):
            self.register_buffer(
                f"fixed_{index}", perception_kernel(name, config.radius)
            )
        for index in range(self.learnable_count):
            kernel = perception_kernel(
                config.learnable_kernel_init,
                config.radius,
                config.random_kernel_seed + index,
            )
            self.register_parameter(f"learnable_{index}", nn.Parameter(kernel))

    @property
    def kernel_count(self) -> int:
        return len(self.fixed_names) + self.learnable_count

    def kernel_bank(self) -> torch.Tensor:
        kernels = [
            getattr(self, f"fixed_{index}") for index in range(len(self.fixed_names))
        ]
        kernels.extend(
            getattr(self, f"learnable_{index}") for index in range(self.learnable_count)
        )
        return torch.stack(kernels)

    def filters(self) -> torch.Tensor:
        kernels = self.kernel_bank()
        return (
            kernels.unsqueeze(0)
            .expand(self.channels, -1, -1)
            .reshape(self.channels * self.kernel_count, 1, 2 * self.radius + 1)
        )

    def forward(
        self, state: torch.Tensor, filters: torch.Tensor | None = None
    ) -> torch.Tensor:
        filters = self.filters() if filters is None else filters
        return F.conv1d(state, filters, padding=self.radius, groups=self.channels)


class UpdateRule(nn.Module):
    def __init__(self, config: ModelConfig, perception_channels: int):
        super().__init__()
        self.channels = config.channels
        self.activation = config.activation
        self.gate = config.gate
        output_channels = (
            config.channels if config.gate == "none" else 2 * config.channels
        )
        self.hidden = nn.Conv1d(perception_channels, config.hidden_size, 1)
        self.output = nn.Conv1d(
            config.hidden_size,
            output_channels,
            1,
            bias=config.gate != "none",
        )
        nn.init.zeros_(self.output.weight[: config.channels])
        if config.gate != "none":
            nn.init.zeros_(self.output.bias[: config.channels])
            nn.init.zeros_(self.output.weight[config.channels :])
            nn.init.constant_(self.output.bias[config.channels :], config.gate_bias)

    def forward(self, perception: torch.Tensor) -> torch.Tensor:
        return self.from_hidden(self.hidden(perception))

    def from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.activation == "relu":
            hidden = F.relu(hidden)
        elif self.activation == "softplus":
            hidden = F.softplus(hidden)
        else:
            raise RuntimeError(f"invalid activation configured: {self.activation}")
        output = self.output(hidden)
        if self.gate == "none":
            return output
        delta, gate_values = output.split(self.channels, dim=1)
        if self.gate == "linear":
            gate = gate_values
        elif self.gate == "sigmoid":
            gate = torch.sigmoid(gate_values)
        elif self.gate == "tanh":
            gate = torch.tanh(gate_values)
        elif self.gate == "relu":
            gate = F.relu(gate_values)
        else:
            raise RuntimeError(f"invalid gate configured: {self.gate}")
        return delta * gate


class NeuralCellularAutomaton(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        task_names: tuple[str, ...],
    ):
        super().__init__()
        config.validate()
        if not task_names or any(
            not isinstance(name, str) or not name.strip() for name in task_names
        ):
            raise ValueError("task names must be non-empty strings")
        if len(set(task_names)) != len(task_names):
            raise ValueError("task names must be unique")
        self.config = config
        self.task_names = tuple(task_names)
        shape = (len(task_names), config.program_length, config.program_channels)
        if config.program_mode == "zero":
            self.register_buffer("programs", torch.zeros(shape))
        else:
            self.programs = nn.Parameter(torch.empty(shape))
            nn.init.normal_(self.programs, std=config.program_init_std)
        self.perception = Perception(config)
        self.rule = UpdateRule(config, config.channels * self.perception.kernel_count)
        update_mask = torch.ones(1, config.channels, 1)
        if not config.program_mutable:
            update_mask[:, : config.program_channels] = 0.0
        self.register_buffer("update_mask", update_mask)
        self.register_buffer("mutable_mask", update_mask.bool())

    @property
    def device(self) -> torch.device:
        return next(self.rule.parameters()).device

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def shared_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(
            parameter
            for name, parameter in self.named_parameters()
            if name != "programs"
        )

    @property
    def program_parameters(self) -> tuple[nn.Parameter, ...]:
        return (self.programs,) if isinstance(self.programs, nn.Parameter) else ()

    def task_index(self, task_name: str) -> int:
        try:
            return self.task_names.index(task_name)
        except ValueError as error:
            raise ValueError(f"unknown task: {task_name!r}") from error

    def program_tape(self, task_indices: torch.Tensor, tape_slots: int) -> torch.Tensor:
        if type(tape_slots) is not int or tape_slots < self.config.program_length:
            raise ValueError("tape must contain the complete program")
        task_indices = torch.as_tensor(task_indices, dtype=torch.int64)
        if task_indices.ndim != 1:
            raise ValueError("task_indices must be one-dimensional")
        if bool(((task_indices < 0) | (task_indices >= len(self.task_names))).any()):
            raise ValueError("task index is out of range")
        task_indices = task_indices.to(self.device)
        positions = torch.arange(tape_slots, device=self.device)
        phases = positions.remainder(self.config.program_length)
        program = self.programs[task_indices][:, phases].transpose(1, 2)
        if self.config.program_placement == "prefix":
            program = program * (positions < self.config.program_length)
        return program

    def initial_state(
        self, input_tape: torch.Tensor, task_indices: torch.Tensor
    ) -> torch.Tensor:
        if input_tape.ndim != 2 or min(input_tape.shape) < 1:
            raise ValueError("input_tape must have shape (batch, tape_slots)")
        if not torch.is_floating_point(input_tape):
            raise ValueError("input_tape must be floating point")
        if input_tape.device != self.device:
            raise ValueError("input tape and model must be on the same device")
        if input_tape.dtype != self.rule.hidden.weight.dtype:
            raise ValueError("input tape and model must have the same dtype")
        task_indices = torch.as_tensor(task_indices, dtype=torch.int64)
        if task_indices.ndim != 1 or task_indices.shape[0] != input_tape.shape[0]:
            raise ValueError("one task index is required per input")
        state = torch.zeros(
            input_tape.shape[0],
            self.config.channels,
            *input_tape.shape[1:],
            device=input_tape.device,
            dtype=input_tape.dtype,
        )
        state[:, : self.config.program_channels] = self.program_tape(
            task_indices, input_tape.shape[-1]
        ).to(dtype=input_tape.dtype)
        state[:, self.config.io_channel] = input_tape
        return state

    def _step(
        self,
        state: torch.Tensor,
        perception_noise_std: float,
        hidden_filters: torch.Tensor,
        fire_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        hidden = F.conv1d(
            state, hidden_filters, self.rule.hidden.bias, padding=self.config.radius
        )
        if self.training and perception_noise_std > 0:
            # Noise lives in perception space, before its hidden projection.
            noise = (
                torch.randn(
                    state.shape[0],
                    self.config.channels * self.perception.kernel_count,
                    state.shape[-1],
                    device=state.device,
                    dtype=state.dtype,
                )
                * perception_noise_std
            )
            hidden = hidden + F.conv1d(noise, self.rule.hidden.weight)
        delta = self.rule.from_hidden(hidden)
        if self.config.state_leak:
            delta = delta - self.config.state_leak * state
        delta = delta * self.update_mask
        if fire_mask is not None:
            delta = delta * fire_mask
        updated = state + delta
        if self.config.max_abs_state is not None:
            updated = updated.clamp(
                -self.config.max_abs_state, self.config.max_abs_state
            )
        return torch.where(self.mutable_mask, updated, state)

    def hidden_filters(self) -> torch.Tensor:
        weights = self.rule.hidden.weight.reshape(
            self.config.hidden_size, self.config.channels, self.perception.kernel_count
        )
        return torch.einsum("hck,kw->hcw", weights, self.perception.kernel_bank())

    def _validate_state(self, state: torch.Tensor, noise: float) -> None:
        if (
            state.ndim != 3
            or state.shape[1] != self.config.channels
            or min(state.shape) < 1
        ):
            raise ValueError(
                f"state must have shape (batch, {self.config.channels}, tape_slots)"
            )
        if not torch.is_floating_point(state) or state.device != self.device:
            raise ValueError("state must be floating point and on the model device")
        if state.dtype != self.rule.hidden.weight.dtype:
            raise ValueError("state and model must have the same dtype")
        if state.shape[-1] < self.config.program_length:
            raise ValueError("tape must contain the complete program")
        if (
            isinstance(noise, bool)
            or not isinstance(noise, (int, float))
            or not math.isfinite(noise)
            or noise < 0
        ):
            raise ValueError("perception_noise_std must be finite and non-negative")

    def step(
        self, state: torch.Tensor, perception_noise_std: float = 0.0
    ) -> torch.Tensor:
        self._validate_state(state, perception_noise_std)
        fire_mask = None
        if self.config.fire_rate < 1.0:
            fire_mask = (
                torch.rand(state.shape[0], 1, *state.shape[2:], device=state.device)
                < self.config.fire_rate
            )
        return self._step(
            state,
            perception_noise_std,
            self.hidden_filters(),
            fire_mask,
        )

    def forward(
        self,
        initial_state: torch.Tensor,
        steps: int,
        perception_noise_std: float = 0.0,
        *,
        io_only: bool = False,
        start_step: int = 0,
    ) -> torch.Tensor:
        if type(steps) is not int or steps < 0:
            raise ValueError("steps must be a non-negative integer")
        self._validate_state(initial_state, perception_noise_std)
        if type(io_only) is not bool:
            raise TypeError("io_only must be a boolean")
        if type(start_step) is not int or not 0 <= start_step <= steps:
            raise ValueError("start_step must be in [0, steps]")
        states = []
        state = initial_state
        if start_step == 0:
            states.append(state[:, self.config.io_channel] if io_only else state)
        hidden_filters = self.hidden_filters()
        fire_masks = None
        if self.config.fire_rate < 1.0 and steps:
            fire_masks = (
                torch.rand(
                    steps,
                    state.shape[0],
                    1,
                    *state.shape[2:],
                    device=state.device,
                )
                < self.config.fire_rate
            )
        for index in range(steps):
            fire_mask = None if fire_masks is None else fire_masks[index]
            state = self._step(
                state,
                perception_noise_std,
                hidden_filters,
                fire_mask,
            )
            if index + 1 >= start_step:
                states.append(state[:, self.config.io_channel] if io_only else state)
        return torch.stack(states, dim=1)
