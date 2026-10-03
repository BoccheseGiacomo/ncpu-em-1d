import ast
import json
from dataclasses import replace
from pathlib import Path

import matplotlib
import pytest
import torch
from PIL import Image

from ncpu_computer_1d import (
    TapeLayout,
    TestCase,
    encode_strings,
    evolution_phase,
    interpret_tape,
    quantize,
    rollout_rgb,
    save_gif,
)
from test_training import tiny_config


matplotlib.use("Agg")
ROOT = Path(__file__).resolve().parents[1]


def test_codec_strict_thresholds_and_interpretation():
    assert quantize(torch.tensor([-0.334, -0.333, 0, 0.333, 0.334])).tolist() == [
        -1,
        0,
        0,
        0,
        1,
    ]
    values = encode_strings(("101B11BB0",), 9)[0]
    assert interpret_tape(values, "multiple").binary_strings == ("101", "11")
    assert interpret_tape(values, "single").binary_strings == ("101",)
    assert interpret_tape(encode_strings(("1011",), 6)[0]).integers == (11,)
    with pytest.raises(ValueError):
        quantize(torch.tensor([float("nan")]))


@pytest.mark.parametrize("placement", ["periodic", "prefix"])
def test_gif_all_channels_and_static_program(tmp_path, placement):
    config = tiny_config(program_length=3, program_placement=placement)
    frames = torch.zeros(4, config.model.channels, 6)
    io = config.model.io_channel
    frames[:, io] = encode_strings(("010",), 6)
    rgb = rollout_rgb(frames)
    assert rgb.shape == (4, config.model.channels, 6, 3)
    frames[:, -1] = 1000
    changed = rollout_rgb(frames)
    torch.testing.assert_close(rgb[:, :-1], changed[:, :-1])
    assert changed[0, -1, 0].tolist() == [180, 4, 38]
    assert rgb[0, io, 0].tolist() == [59, 76, 192]
    assert rgb[0, io, 1].tolist() == [180, 4, 38]
    assert rgb[0, io, -1].tolist() == [245, 245, 245]
    path = save_gif(
        frames,
        tmp_path / "demo.gif",
        layout=TapeLayout(6),
        config=config,
        program=torch.tensor([[0.1, -0.2], [0.2, 0.3], [0.4, -0.1]]),
        task_name="reverse",
        input_symbols="010",
        target_symbols="010",
        free_steps=1,
        supervision_steps=2,
    )
    with Image.open(path) as image:
        assert image.n_frames == 4
        assert image.width >= 6 * 36
        assert image.height > config.model.channels * 36
        image.seek(0)
        first = image.convert("RGB")
        first.save(tmp_path / "preview.png")
        for channel in range(config.model.channels):
            expected = tuple(changed[0, channel, 0].tolist())
            assert first.getpixel((76 + 18, 116 + channel * 36 + 18)) == expected
    assert evolution_phase(0, 1, 2) == "Free evolution"
    assert evolution_phase(2, 1, 2) == "Supervision window"
    assert evolution_phase(4, 1, 2) == "Beyond training window"


@pytest.mark.parametrize("placement", ["periodic", "prefix"])
def test_complete_notebook_initialization_and_tiny_execution(
    monkeypatch, tmp_path, placement
):
    monkeypatch.chdir(ROOT / "notebooks")
    book = json.loads((ROOT / "notebooks/run.ipynb").read_text(encoding="utf-8"))
    namespace = {"__name__": "__main__"}

    def execute(index, **overrides):
        module = ast.parse("".join(book["cells"][index]["source"]))
        for node in module.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name) and target.id in overrides:
                    key = "_test_" + target.id
                    namespace[key] = overrides[target.id]
                    node.value = ast.Name(id=key, ctx=ast.Load())
        exec(compile(ast.fix_missing_locations(module), "notebook", "exec"), namespace)

    # Run the real first cell, including its large fixed-test domain checks.
    for cell in book["cells"]:
        compile("".join(cell["source"]), "notebook", "exec")
    execute(0)
    assert namespace["config"].model is namespace["model_config"]
    assert namespace["config"].training is namespace["training_config"]
    assert namespace["config"].training.updates == namespace["UPDATES"]
    assert namespace["config"].model.program_length == namespace["PROGRAM_LENGTH"]
    assert namespace["config"].model.program_placement == namespace["PROGRAM_PLACEMENT"]
    assert namespace["config"].test_cases == namespace["TEST_CASES"]
    assert namespace["report"].parameter_count > 0
    # Exercise all workflow cells without launching the default training run.
    config = tiny_config(program_length=3, program_placement=placement)
    namespace.update(
        config=config,
        model_config=config.model,
        training_config=config.training,
        UPDATES=2,
        PROGRAM_CHANNELS=2,
        PROGRAM_LENGTH=3,
        PROGRAM_PLACEMENT=placement,
        TASKS=(("reverse", 2.0),),
        TASK_NAMES=("reverse",),
        checkpoint_dir=tmp_path,
        checkpoint_path=tmp_path / "best.pt",
        TEST_CASES=(TestCase("tiny", 6, 2, 1, 2),),
        EVALUATION_BATCH_SIZE=4,
    )

    class DisplayHandle:
        def update(self, value):
            if hasattr(value, "canvas"):
                value.canvas.draw()

    def display(value, **kwargs):
        handle = DisplayHandle()
        handle.update(value)
        return handle

    namespace["display"] = display
    execute(1, PROGRESS_EVERY=1)
    execute(2)
    execute(
        3,
        VISUALIZATION_INPUT="10",
        VISUALIZATION_TAPE_SLOTS=6,
        VISUALIZATION_FREE_STEPS=1,
        VISUALIZATION_SUPERVISION_STEPS=2,
        GIF_PATH=tmp_path / "output.gif",
    )
    execute(4)
    assert (tmp_path / "output.gif").is_file()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("radius", [1, 2])
def test_cuda_training_validation_and_resume(tmp_path, radius):
    from ncpu_computer_1d import Trainer

    config = tiny_config(radius=radius, program_length=3, program_placement="prefix")
    config = replace(config, training=replace(config.training, device="cuda"))
    trainer = Trainer(config, (("reverse", 2.0), ("copy", 1.0)))
    trainer.train_step()
    before = torch.cuda.get_rng_state().clone()
    result = trainer.validation_metrics()
    assert torch.cuda.get_rng_state().equal(before)
    assert set(result.task_accuracies) == {"reverse", "copy"}
    path = tmp_path / "cuda.pt"
    trainer.save(path)
    first = trainer.train_step()
    state = {name: value.clone() for name, value in trainer.model.state_dict().items()}
    resumed = Trainer.from_checkpoint(path)
    second = resumed.train_step()
    assert first.to_dict() == second.to_dict()
    for name, value in state.items():
        torch.testing.assert_close(
            value, resumed.model.state_dict()[name], rtol=0, atol=0
        )
