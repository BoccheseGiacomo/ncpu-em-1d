from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

from .config import ExperimentConfig
from .tape import TapeLayout, interpret_tape, tensor_to_symbols, validate_symbols


_NEGATIVE = (59, 76, 192)
_NEUTRAL = (245, 245, 245)
_POSITIVE = (180, 4, 38)


def evolution_phase(step: int, free_steps: int, supervision_steps: int) -> str:
    if any(type(v) is not int for v in (step, free_steps, supervision_steps)):
        raise TypeError("evolution times must be integers")
    if step < 0 or free_steps < 0 or supervision_steps < 1:
        raise ValueError("evolution times are invalid")
    if step <= free_steps:
        return "Free evolution"
    if step <= free_steps + supervision_steps:
        return "Supervision window"
    return "Beyond training window"


def _colorize(values: torch.Tensor) -> torch.Tensor:
    values = values.clamp(-1, 1)
    neutral = values.new_tensor(_NEUTRAL)
    endpoint = torch.where(
        (values < 0).unsqueeze(-1),
        values.new_tensor(_NEGATIVE),
        values.new_tensor(_POSITIVE),
    )
    return (
        (neutral + values.abs().unsqueeze(-1) * (endpoint - neutral))
        .round()
        .to(torch.uint8)
    )


def rollout_rgb(rollout: torch.Tensor) -> torch.Tensor:
    if rollout.ndim != 3 or min(rollout.shape) < 1:
        raise ValueError("rollout must have shape (time, channels, tape_slots)")
    if not torch.is_floating_point(rollout) or not torch.isfinite(rollout).all():
        raise ValueError("rollout must contain finite floating-point states")
    return _colorize(rollout)


def save_gif(
    rollout: torch.Tensor,
    path: str | Path,
    *,
    layout: TapeLayout,
    config: ExperimentConfig,
    program: torch.Tensor,
    task_name: str,
    input_symbols: str,
    target_symbols: str,
    free_steps: int,
    supervision_steps: int,
    output_mode: str = "single",
    duration_ms: int = 80,
    scale: int = 36,
) -> Path:
    config.validate()
    evolution_phase(0, free_steps, supervision_steps)
    if any(type(v) is not int or v < 1 for v in (duration_ms, scale)):
        raise ValueError("duration_ms and scale must be positive integers")
    if output_mode not in {"single", "multiple"} or not task_name.strip():
        raise ValueError("invalid output mode or task name")
    for symbols in (input_symbols, target_symbols):
        validate_symbols(symbols, allow_empty=True)
        if len(symbols) > layout.tape_slots - 1:
            raise ValueError("visualized input and output must leave a blank")
    frames = rollout_rgb(rollout.detach().cpu())
    if rollout.shape[1:] != (config.model.channels, layout.tape_slots):
        raise ValueError("rollout dimensions do not match model and tape")
    if layout.tape_slots < config.model.program_length:
        raise ValueError("visualized tape cannot contain the complete program")
    if program.shape != (config.model.program_length, config.model.program_channels):
        raise ValueError("program has the wrong shape")
    if not torch.isfinite(program).all():
        raise ValueError("program must be finite")
    font = ImageFont.load_default()
    foreground = "#e7ebef"
    left, tape_y = 76, 116
    width = max(580, 2 * left + layout.tape_slots * scale)
    grid_height = config.model.channels * scale
    program_y = tape_y + grid_height + 56
    program_values = program.detach().float().cpu().T
    program_image = Image.fromarray(_colorize(program_values).numpy()).resize(
        (config.model.program_length * scale, config.model.program_channels * scale),
        Image.Resampling.NEAREST,
    )
    width = max(width, 2 * left + program_image.width)
    values = rollout.detach().cpu()[:, config.model.io_channel]
    images = []
    for step, frame in enumerate(frames):
        image = Image.new(
            "RGB", (width, program_y + program_image.height + 28), "#161a1f"
        )
        draw = ImageDraw.Draw(image)
        interpreted = interpret_tape(values[step], output_mode)
        lines = (
            f"t={step}/{len(frames)-1}   "
            f"{evolution_phase(step, free_steps, supervision_steps)}",
            f"Task: {task_name}   Input: {input_symbols or '<empty>'}",
            f"Target: {target_symbols or '<empty>'}",
            f"Raw: {tensor_to_symbols(values[step])}   "
            f"Decoded: {' | '.join(interpreted.binary_strings)}",
            "All state channels; I/O highlighted; colors clipped to [-1,+1]",
        )
        for row, text in enumerate(lines):
            draw.text((left, 6 + 19 * row), text, font=font, fill=foreground)
        tape = Image.fromarray(frame.numpy()).resize(
            (layout.tape_slots * scale, grid_height), Image.Resampling.NEAREST
        )
        image.paste(tape, (left, tape_y))
        for index in range(layout.tape_slots):
            x = left + index * scale
            draw.text((x + 3, tape_y - 14), str(index), font=font, fill=foreground)
        for channel in range(config.model.channels):
            y = tape_y + channel * scale
            if channel < config.model.program_channels:
                label = f"{channel}: P{channel}"
            elif channel == config.model.io_channel:
                label = f"{channel}: I/O"
            else:
                label = f"{channel}: C{channel - config.model.io_channel - 1}"
            color = "#e8bd36" if channel == config.model.io_channel else foreground
            draw.text((8, y + scale // 2 - 5), label, font=font, fill=color)
            for index in range(layout.tape_slots):
                x = left + index * scale
                draw.rectangle((x, y, x + scale - 1, y + scale - 1), outline="#555555")
            if channel == config.model.io_channel:
                draw.rectangle(
                    (left, y, left + layout.tape_slots * scale - 1, y + scale - 1),
                    outline=color,
                    width=2,
                )
        # One fixed scale makes amplitudes comparable across frames and channels.
        ramp = torch.linspace(-1, 1, 180)
        bar = Image.fromarray(_colorize(ramp).numpy()[None]).resize((180, 8))
        image.paste(bar, (left, tape_y + grid_height + 12))
        for offset, label, alignment in ((0, "-1", 0), (90, "0", 0.5), (180, "+1", 1)):
            x = left + offset - alignment * draw.textlength(label, font=font)
            draw.text((x, tape_y + grid_height + 23), label, font=font, fill=foreground)
        draw.text(
            (left, program_y - 18),
            f"Initial program: {config.model.program_placement} "
            "(columns=phase, rows=channel; colors clipped to [-1,1])",
            font=font,
            fill=foreground,
        )
        image.paste(program_image, (left, program_y))
        for channel, row in enumerate(program_values.tolist()):
            for phase, value in enumerate(row):
                draw.text(
                    (left + phase * scale + 2, program_y + channel * scale + 4),
                    f"{value:.2f}",
                    font=font,
                    fill="#161a1f",
                )
        images.append(image)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        path, save_all=True, append_images=images[1:], duration=duration_ms, loop=0
    )
    return path
