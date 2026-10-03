from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from ncpu_computer_1d import ModelConfig, NeuralCellularAutomaton, encode_strings
from ncpu_computer_1d.model import Perception, perception_kernel


@pytest.mark.parametrize("radius", [1, 2])
@pytest.mark.parametrize("length", [1, 2, 7])
def test_perception_outputs_and_gradients_match_explicit_windows(radius, length):
    config = ModelConfig(program_channels=1, computation_channels=1, radius=radius)
    perception = Perception(config).double()
    x = torch.randn(2, config.channels, length, dtype=torch.float64, requires_grad=True)
    kernels = perception.kernel_bank()
    padded = F.pad(x, (radius, radius))
    reference = torch.stack(
        [
            torch.stack(
                [
                    (padded[:, channel, pos : pos + 2 * radius + 1] * kernel).sum(-1)
                    for pos in range(length)
                ],
                dim=-1,
            )
            for channel in range(config.channels)
            for kernel in kernels
        ],
        dim=1,
    )
    actual = perception(x)
    torch.testing.assert_close(actual, reference)
    params = (x, *perception.parameters())
    actual_grads = torch.autograd.grad(actual.square().sum(), params, retain_graph=True)
    reference_grads = torch.autograd.grad(reference.square().sum(), params)
    for actual_grad, reference_grad in zip(actual_grads, reference_grads):
        torch.testing.assert_close(actual_grad, reference_grad)


def test_exact_banks_parameter_counts_and_seeded_learned_kernels():
    for radius, hidden, expected in [(1, 96, 9315), (2, 64, 9290), (2, 96, 13930)]:
        config = ModelConfig(radius=radius, hidden_size=hidden)
        model = NeuralCellularAutomaton(config, ("reverse",))
        assert sum(p.numel() for p in model.shared_parameters) == expected
        assert model.perception.learnable_count == radius
        assert model.perception.filters().shape[-1] == 2 * radius + 1
        bank = model.perception.kernel_bank()
        torch.testing.assert_close(bank[2], bank[3] - bank[0])
    assert perception_kernel("neighbor_contrast", 2).tolist() == [
        -0.5,
        0.5,
        0,
        0.5,
        -0.5,
    ]
    random_config = ModelConfig(radius=2, learnable_kernel_init="random")
    a, b = Perception(random_config), Perception(random_config)
    torch.testing.assert_close(a.kernel_bank(), b.kernel_bank())
    assert not torch.equal(a.learnable_0, a.learnable_1)


@pytest.mark.parametrize("activation", ["relu", "softplus"])
@pytest.mark.parametrize("gate", ["none", "linear", "sigmoid", "tanh", "relu"])
def test_activation_gating_and_evolution_reference(activation, gate):
    config = ModelConfig(
        program_channels=1,
        computation_channels=2,
        hidden_size=5,
        activation=activation,
        gate=gate,
        fire_rate=1.0,
        state_leak=0.02,
        max_abs_state=None,
    )
    model = NeuralCellularAutomaton(config, ("reverse",))
    with torch.no_grad():
        model.rule.output.weight.normal_(std=0.1)
    x = torch.randn(2, config.channels, 4, requires_grad=True)
    features = model.perception(x)
    hidden = model.rule.hidden(features)
    hidden = F.relu(hidden) if activation == "relu" else F.softplus(hidden)
    output = model.rule.output(hidden)
    if gate == "none":
        delta = output
    else:
        delta, g = output.chunk(2, dim=1)
        functions = {
            "linear": lambda y: y,
            "sigmoid": torch.sigmoid,
            "tanh": torch.tanh,
            "relu": F.relu,
        }
        delta = delta * functions[gate](g)
    expected = x + model.update_mask * (delta - config.state_leak * x)
    actual = model.step(x)
    torch.testing.assert_close(actual, expected)
    variables = (x, *model.shared_parameters)
    grad_a = torch.autograd.grad(actual.square().sum(), variables, retain_graph=True)
    grad_b = torch.autograd.grad(expected.square().sum(), variables)
    for a, b in zip(grad_a, grad_b):
        torch.testing.assert_close(a, b)


@pytest.mark.parametrize("mode", ["zero", "learned_read_only", "learned_mutable"])
def test_masked_leak_and_frozen_channels_through_clipping(mode):
    config = ModelConfig(program_mode=mode, state_leak=0.1, max_abs_state=0.5)
    model = NeuralCellularAutomaton(config, ("copy",))
    state = torch.full((2, config.channels, 4), 0.4)
    state[:, : config.program_channels] = 2.0
    mask = torch.tensor([[[True, False, True, False]]]).expand(2, -1, -1)
    actual = model._step(state, 0, model.hidden_filters(), mask)
    expected = (state - 0.1 * state * model.update_mask * mask).clamp(-0.5, 0.5)
    expected = torch.where(model.mutable_mask, expected, state)
    torch.testing.assert_close(actual, expected)
    assert torch.equal(
        actual[:, config.io_channel, 1::2], state[:, config.io_channel, 1::2]
    )


def test_program_broadcast_and_gradients_survive_frozen_clipping():
    config = ModelConfig(
        program_channels=2, computation_channels=1, fire_rate=1.0, max_abs_state=0.5
    )
    model = NeuralCellularAutomaton(config, ("copy", "reverse"))
    with torch.no_grad():
        model.programs.copy_(torch.tensor([[[2.0, 3.0]], [[4.0, 5.0]]]))
        model.rule.hidden.weight.fill_(0.01)
        model.rule.hidden.bias.fill_(0.1)
        model.rule.output.weight.fill_(0.001)
    inputs = torch.zeros(2, 5)
    state = model.initial_state(inputs, torch.tensor([1, 0]))
    torch.testing.assert_close(
        state[:, :2], model.programs[[1, 0], 0, :, None].expand(-1, -1, 5)
    )
    trajectory = model(state, 3)
    torch.testing.assert_close(
        trajectory[:, :, :2], state[:, None, :2].expand(-1, 4, -1, -1)
    )
    trajectory[:, -1, config.io_channel].sum().backward()
    assert bool((model.programs.grad.abs().sum((1, 2)) > 0).all())
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.shared_parameters
    )


def test_firing_is_fresh_reproducible_and_step_agrees_with_rollout():
    config = ModelConfig(fire_rate=0.5, state_leak=0.1, max_abs_state=None)
    model = NeuralCellularAutomaton(config, ("copy",))
    initial = torch.ones(8, config.channels, 19)
    torch.manual_seed(42)
    a = model(initial, 4)
    b = model(initial, 4)
    assert not torch.equal(a, b)
    torch.manual_seed(42)
    torch.testing.assert_close(a, model(initial, 4), rtol=0, atol=0)
    torch.manual_seed(71)
    single = model.step(initial)
    torch.manual_seed(71)
    torch.testing.assert_close(single, model(initial, 1)[:, 1], rtol=0, atol=0)
    assert torch.equal(model(initial, 0)[:, 0], initial)


def test_identity_without_leak_and_invalid_entry_arguments():
    model = NeuralCellularAutomaton(ModelConfig(state_leak=0), ("copy",))
    initial = model.initial_state(encode_strings(("01", ""), 4), torch.tensor([0, 0]))
    torch.testing.assert_close(
        model(initial, 3), initial[:, None].expand(-1, 4, -1, -1)
    )
    for noise in [-1, float("nan"), float("inf")]:
        with pytest.raises(ValueError):
            model(initial, 1, noise)
    with pytest.raises(ValueError):
        model(torch.zeros(2, 3, 4), 0)
    with pytest.raises(ValueError):
        model(initial, -1)
    with pytest.raises(ValueError):
        model.initial_state(torch.zeros(1, 4), torch.tensor([1]))
    with pytest.raises(ValueError):
        replace(model.config, radius=3).validate()


def unfused_rollout(model, initial, steps, noise=0):
    state = initial
    states = [state]
    filters = model.perception.filters()
    masks = None
    if model.config.fire_rate < 1:
        masks = (
            torch.rand(steps, state.shape[0], 1, state.shape[-1], device=state.device)
            < model.config.fire_rate
        )
    for step in range(steps):
        perceived = model.perception(state, filters)
        if model.training and noise:
            perceived = perceived + torch.randn_like(perceived) * noise
        delta = model.rule(perceived) - model.config.state_leak * state
        delta = delta * model.update_mask
        if masks is not None:
            delta = delta * masks[step]
        updated = state + delta
        if model.config.max_abs_state is not None:
            updated = updated.clamp(
                -model.config.max_abs_state, model.config.max_abs_state
            )
        state = torch.where(model.mutable_mask, updated, state)
        states.append(state)
    return torch.stack(states, dim=1)


@pytest.mark.parametrize("radius", [1, 2])
@pytest.mark.parametrize("activation", ["relu", "softplus"])
@pytest.mark.parametrize("gate", ["none", "linear", "sigmoid", "tanh", "relu"])
@pytest.mark.parametrize("noise", [0, 0.01])
def test_fusion_multistep_outputs_and_all_gradients(radius, activation, gate, noise):
    config = ModelConfig(
        program_channels=2,
        computation_channels=1,
        hidden_size=6,
        program_length=3,
        radius=radius,
        activation=activation,
        gate=gate,
        fire_rate=0.7,
        max_abs_state=0.6,
        learnable_kernel_init="random",
    )
    model = NeuralCellularAutomaton(config, ("copy", "reverse")).double()
    with torch.no_grad():
        model.rule.output.weight.normal_(std=0.1)
        model.programs[0, 0, 0] = 2  # Frozen program must survive clipping.
    inputs = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    initial = model.initial_state(inputs, torch.tensor([0, 1]))
    torch.manual_seed(92)
    actual = model(initial, 4, noise)
    torch.manual_seed(92)
    expected = unfused_rollout(model, initial, 4, noise)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-11)
    variables = (inputs, *model.parameters())
    a = torch.autograd.grad(actual.square().sum(), variables, retain_graph=True)
    b = torch.autograd.grad(expected.square().sum(), variables)
    for left, right in zip(a, b):
        torch.testing.assert_close(left, right, rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize("placement", ["periodic", "prefix"])
@pytest.mark.parametrize("mode", ["zero", "learned_read_only", "learned_mutable"])
def test_program_placement_containment_and_phase_gradients(placement, mode):
    config = ModelConfig(
        program_channels=2,
        program_length=3,
        program_placement=placement,
        program_mode=mode,
        hidden_size=5,
        computation_channels=1,
        fire_rate=1,
    )
    model = NeuralCellularAutomaton(config, ("copy", "reverse"))
    if mode != "zero":
        with torch.no_grad():
            model.programs.copy_(torch.arange(12).reshape(2, 3, 2) / 10)
    indices = torch.tensor([1, 0])
    program = model.program_tape(indices, 8)
    expected = model.programs[indices][:, torch.arange(8) % 3].transpose(1, 2).clone()
    if placement == "prefix":
        expected[:, :, 3:] = 0
    torch.testing.assert_close(program, expected, rtol=0, atol=0)
    state = model.initial_state(torch.zeros(2, 8), indices)
    evolved = model(state, 2)
    if mode != "learned_mutable":
        torch.testing.assert_close(
            evolved[:, :, :2], program[:, None].expand(-1, 3, -1, -1), rtol=0, atol=0
        )
    if mode != "zero":
        program.sum().backward()
        assert bool((model.programs.grad.abs().sum(-1) > 0).all())
    with pytest.raises(ValueError, match="complete program"):
        model.initial_state(torch.zeros(1, 2), torch.tensor([0]))
    with pytest.raises(ValueError, match="complete program"):
        model(torch.zeros(1, config.channels, 2), 0)


@pytest.mark.parametrize("placement", ["periodic", "prefix"])
def test_compact_trajectory_retains_full_bptt_and_noise(placement):
    model = NeuralCellularAutomaton(
        ModelConfig(
            program_length=2,
            program_placement=placement,
            program_mode="learned_mutable",
            computation_channels=1,
            hidden_size=6,
        ),
        ("reverse",),
    ).double()
    with torch.no_grad():
        model.rule.output.weight.normal_(std=0.02)
    inputs = torch.randn(2, 5, dtype=torch.float64, requires_grad=True)
    state = model.initial_state(inputs, torch.zeros(2, dtype=torch.long))
    torch.manual_seed(7)
    full = model(state, 7, 0.01)[:, 4:, model.config.io_channel]
    torch.manual_seed(7)
    compact = model(state, 7, 0.01, io_only=True, start_step=4)
    torch.testing.assert_close(compact, full, rtol=0, atol=0)
    variables = (inputs, *model.parameters())
    a = torch.autograd.grad(full.square().mean(), variables, retain_graph=True)
    b = torch.autograd.grad(compact.square().mean(), variables)
    for left, right in zip(a, b):
        torch.testing.assert_close(left, right, rtol=1e-12, atol=1e-13)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("radius", [1, 2])
@pytest.mark.parametrize("noise", [0, 0.01])
def test_cuda_fp32_fusion_outputs_gradients_and_eval_noise(radius, noise):
    config = ModelConfig(
        program_channels=2,
        program_length=3,
        program_placement="prefix",
        program_mode="learned_mutable",
        hidden_size=8,
        computation_channels=2,
        radius=radius,
        gate="sigmoid",
        activation="softplus",
        fire_rate=0.8,
    )
    model = NeuralCellularAutomaton(config, ("copy", "reverse")).cuda()
    with torch.no_grad():
        model.rule.output.weight.normal_(std=0.03)
    inputs = torch.randn(2, 4, device="cuda", requires_grad=True)
    state = model.initial_state(inputs, torch.tensor([0, 1]))
    torch.manual_seed(29)
    actual = model(state, 5, noise)
    torch.manual_seed(29)
    expected = unfused_rollout(model, state, 5, noise)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    variables = (inputs, *model.parameters())
    a = torch.autograd.grad(actual.square().sum(), variables, retain_graph=True)
    b = torch.autograd.grad(expected.square().sum(), variables)
    for left, right in zip(a, b):
        torch.testing.assert_close(left, right, rtol=5e-5, atol=2e-5)
    model.eval()
    torch.manual_seed(9)
    without_noise = model(state, 2)
    torch.manual_seed(9)
    with_noise = model(state, 2, 10)
    torch.testing.assert_close(without_noise, with_noise, rtol=0, atol=0)


def test_default_program_and_invalid_settings():
    config = ModelConfig()
    assert config.program_length == 1 and config.program_placement == "periodic"
    model = NeuralCellularAutomaton(config, ("reverse", "copy"))
    indices = torch.tensor([1, 0])
    expected = model.programs[indices, 0, :, None].expand(-1, -1, 7)
    torch.testing.assert_close(model.program_tape(indices, 7), expected, rtol=0, atol=0)
    for length in (0, -1, True, 1.5):
        with pytest.raises((ValueError, TypeError)):
            replace(config, program_length=length).validate()
    with pytest.raises(ValueError):
        replace(config, program_placement="other").validate()


def test_mutable_prefix_tail_is_only_zero_at_initialization():
    config = ModelConfig(
        program_channels=2,
        program_length=2,
        program_placement="prefix",
        program_mode="learned_mutable",
        fire_rate=1,
        hidden_size=4,
    )
    model = NeuralCellularAutomaton(config, ("copy",))
    with torch.no_grad():
        model.rule.hidden.weight.zero_()
        model.rule.hidden.bias.fill_(1)
        model.rule.output.weight.fill_(0.01)
    initial = model.initial_state(torch.zeros(1, 5), torch.tensor([0]))
    assert torch.count_nonzero(initial[:, :2, 2:]) == 0
    evolved = model(initial, 1)[:, 1]
    assert bool((evolved[:, :2, 2:] > 0).all())
