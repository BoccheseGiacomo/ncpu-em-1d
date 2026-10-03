import pytest
import torch

from ncpu_computer_1d.tasks import (
    StringExample,
    StringTask,
    MultiTaskDataset,
    TaskDataset,
    addition_task,
    append_one_task,
    append_zero_task,
    binary_tasks,
    bitwise_not_task,
    copy_task,
    parity_task,
    reverse_task,
    semantic_correct,
)


def test_binary_tasks_include_empty_and_all_shorter_strings_in_order():
    reverse = reverse_task(2)
    assert tuple(reverse.examples) == (
        StringExample("", ""),
        StringExample("0", "0"),
        StringExample("1", "1"),
        StringExample("00", "00"),
        StringExample("01", "10"),
        StringExample("10", "01"),
        StringExample("11", "11"),
    )
    bit_not = [
        (example.input, example.target) for example in bitwise_not_task(2).examples
    ]
    assert bit_not == [
        ("", ""),
        ("0", "1"),
        ("1", "0"),
        ("00", "11"),
        ("01", "10"),
        ("10", "01"),
        ("11", "00"),
    ]
    parity = {example.input: example.target for example in parity_task(1).examples}
    assert parity == {"": "0", "0": "0", "1": "1"}
    assert len(addition_task(2).examples) == 16
    assert tuple(copy_task(1).examples) == (
        StringExample("", ""),
        StringExample("0", "0"),
        StringExample("1", "1"),
    )
    assert tuple(append_zero_task(1).examples) == (
        StringExample("", "0"),
        StringExample("0", "00"),
        StringExample("1", "10"),
    )
    assert tuple(append_one_task(1).examples) == (
        StringExample("", "1"),
        StringExample("0", "01"),
        StringExample("1", "11"),
    )


def test_length_preserving_programmed_tasks_have_exact_transformations():
    names = (
        "copy",
        "bit_not",
        "reverse",
        "reverse_not",
        "shift_left_zero",
        "shift_right_zero",
        "gray_encode",
        "prefix_xor",
        "increment",
    )
    tasks = binary_tasks(names, 4, include_shorter=False)
    assert {
        task.name: next(
            example.target for example in task.examples if example.input == "1011"
        )
        for task in tasks
    } == {
        "copy": "1011",
        "bit_not": "0100",
        "reverse": "1101",
        "reverse_not": "0010",
        "shift_left_zero": "0110",
        "shift_right_zero": "0101",
        "gray_encode": "1110",
        "prefix_xor": "1101",
        "increment": "1100",
    }
    assert all(
        tuple(task.examples) == (StringExample("", ""),)
        for task in binary_tasks(names, 0)
    )


def test_programmed_tasks_share_inputs_and_balanced_batches():
    names = (
        "copy",
        "bit_not",
        "reverse",
        "reverse_not",
        "shift_left_zero",
        "shift_right_zero",
        "gray_encode",
        "prefix_xor",
        "increment",
    )
    tasks = binary_tasks(names, 2)
    datasets = MultiTaskDataset.from_tasks(tasks, tape_slots=4)
    assert datasets.task_names == names
    assert all(
        dataset.input_strings == datasets.datasets[0].input_strings
        for dataset in datasets.datasets
    )
    batch = datasets.balanced_sample(3, torch.Generator().manual_seed(4))
    inputs, targets, lengths, task_indices = batch
    assert inputs.shape == targets.shape == (27, 4)
    assert lengths.shape == task_indices.shape == (27,)
    assert torch.bincount(task_indices, minlength=9).tolist() == [3] * 9
    assert set(task_indices.tolist()) == set(range(9))


def test_append_capacity_is_checked_without_automatic_expansion():
    with pytest.raises(ValueError, match="output"):
        TaskDataset.from_task(append_zero_task(2), tape_slots=3)


def test_exact_length_tasks_exclude_empty_and_shorter_strings():
    inputs = [
        example.input for example in reverse_task(2, include_shorter=False).examples
    ]
    assert inputs == ["00", "01", "10", "11"]


def test_task_validation_rejects_ambiguous_targets():
    with pytest.raises(ValueError, match="single-output"):
        StringTask("bad", (StringExample("1", "1B1"),))
    with pytest.raises(ValueError, match="single-B"):
        StringTask("bad", (StringExample("1", "1BB1"),), "multiple")


def test_dataset_uses_direct_values_and_blank_fills_full_capacity():
    task = StringTask("copy", (StringExample("101", "10"), StringExample("", "")))
    dataset = TaskDataset.from_task(task, tape_slots=4)
    inputs, targets, lengths = dataset.take(torch.arange(len(dataset)))
    assert inputs.tolist() == [
        [1.0, -1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]
    assert targets.tolist() == [
        [1.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]
    assert lengths.tolist() == [2, 0]


def test_full_length_input_and_output_must_leave_one_blank():
    with pytest.raises(ValueError, match="input"):
        TaskDataset.from_task(
            StringTask("copy", (StringExample("111", "11"),)), tape_slots=3
        )
    with pytest.raises(ValueError, match="output"):
        TaskDataset.from_task(
            StringTask("append", (StringExample("11", "111"),)), tape_slots=3
        )


def test_semantic_correct_requires_available_blank_but_ignores_later_tail():
    target = torch.tensor([[1, -1, 0, 0]], dtype=torch.int8)
    lengths = torch.tensor([2])
    predictions = torch.tensor([[[1, -1, 0, 1], [1, -1, 1, 0]]], dtype=torch.int8)
    assert semantic_correct(predictions, target, lengths, "single").tolist() == [
        [True, False]
    ]
