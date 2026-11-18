from functools import partial
from pathlib import Path

from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from squid.data.chat_sft import build_assistant_targets, QwenMultiTurnDataset, qwen_multiturn_collate


def test_multiturn_targets_only_supervise_assistant_spans():
    full_ids = list(range(60))

    targets = build_assistant_targets(
        full_ids,
        assistant_spans=[
            (29, 35),
            (53, 59),
        ],
    )

    expected = [-100] * 59
    expected[28:34] = list(range(29, 35))
    expected[52:58] = list(range(53, 59))

    assert targets == expected


def test_qwen_multiturn_dataset_uses_real_template_spans():
    root = Path(__file__).resolve().parents[1]
    model_dir = root / "data/model_pth/qwen3/qwen3-0_6b"
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

    data = [
        {
            "messages": [
                {"role": "system", "content": "你是一个乐于助人的 AI 助手。"},
                {"role": "user", "content": "我叫小吕。"},
                {"role": "assistant", "content": "你好，小吕。"},
                {"role": "user", "content": "那我叫什么？"},
                {"role": "assistant", "content": "你叫小吕。"},
            ]
        }
    ]

    dataset = QwenMultiTurnDataset(data, tokenizer)
    inputs, targets = dataset[0]

    first_response = tokenizer.encode(
        "你好，小吕。<|im_end|>",
        add_special_tokens=False,
    )
    second_response = tokenizer.encode(
        "你叫小吕。<|im_end|>",
        add_special_tokens=False,
    )

    assert len(inputs) == len(targets) == 59
    assert targets[28:34] == first_response
    assert targets[52:58] == second_response

    supervised = set(range(28, 34)) | set(range(52, 58))
    for index, target in enumerate(targets):
        if index in supervised:
            assert target != -100
        else:
            assert target == -100


def test_qwen_multiturn_collate_pads_inputs_but_ignores_padding_targets():
    root = Path(__file__).resolve().parents[1]
    model_dir = root / "data/model_pth/qwen3/qwen3-0_6b"
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

    data = [
        {
            "messages": [
                {"role": "system", "content": "你是一个乐于助人的 AI 助手。"},
                {"role": "user", "content": "我叫小吕。"},
                {"role": "assistant", "content": "你好，小吕。"},
                {"role": "user", "content": "那我叫什么？"},
                {"role": "assistant", "content": "你叫小吕。"},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "你是一个乐于助人的 AI 助手。"},
                {"role": "user", "content": "你好。"},
                {"role": "assistant", "content": "你好。"},
            ]
        },
    ]

    dataset = QwenMultiTurnDataset(data, tokenizer)
    short_input_length = len(dataset[1][0])

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=partial(
            qwen_multiturn_collate,
            pad_token_id=tokenizer.eos_token_id,
        ),
    )

    inputs, targets = next(iter(loader))

    assert inputs.shape == (2, 59)
    assert targets.shape == (2, 59)

    assert (
        inputs[1, short_input_length:] == tokenizer.eos_token_id
    ).all()
    assert (targets[1, short_input_length:] == -100).all()

def test_qwen_cot_dataset_supervises_full_assistant_output():
    root = Path(__file__).resolve().parents[1]
    model_dir = root / "data/model_pth/qwen3/qwen3-0_6b"
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

    data = [
        {
            "messages": [
                {"role": "system", "content": "你是一个乐于助人的 AI 助手。"},
                {"role": "user", "content": "25 * 24 结果是多少？"},
                {
                    "role": "assistant",
                    "content": (
                        "<think>\n"
                        "把 24 拆成 20 + 4。\n"
                        "25 × 20 = 500，25 × 4 = 100。\n"
                        "500 + 100 = 600。\n"
                        "</think>\n\n"
                        "600"
                    ),
                },
            ]
        }
    ]

    dataset = QwenMultiTurnDataset(data, tokenizer)
    inputs, targets = dataset[0]

    expected_output = tokenizer.encode(
        "<think>\n"
        "把 24 拆成 20 + 4。\n"
        "25 × 20 = 500，25 × 4 = 100。\n"
        "500 + 100 = 600。\n"
        "</think>\n\n"
        "600<|im_end|>",
        add_special_tokens=False,
    )

    assert len(inputs) == len(targets) == 94
    assert targets[33:93] == expected_output
    assert targets[:33] == [-100] * 33
    assert targets[93:] == [-100]

    assert targets[33] == tokenizer.convert_tokens_to_ids("<think>")
    assert targets[87] == tokenizer.convert_tokens_to_ids("</think>")
    assert targets[92] == tokenizer.convert_tokens_to_ids("<|im_end|>")

def test_qwen_multiturn_cot_history_matches_template_visibility():
    root = Path(__file__).resolve().parents[1]
    model_dir = root / "data/model_pth/qwen3/qwen3-0_6b"
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

    first_cot = (
        "<think>\n"
        "12 × 16 = 192。\n"
        "</think>\n\n"
        "192"
    )
    second_cot = (
        "<think>\n"
        "原有数量是 192。\n"
        "192 × 3 = 576。\n"
        "</think>\n\n"
        "576"
    )

    messages = [
        {"role": "system", "content": "你是一个乐于助人的 AI 助手。"},
        {
            "role": "user",
            "content": "仓库有 12 包卡片，每包 16 张，库存共有多少张？",
        },
        {"role": "assistant", "content": first_cot},
        {
            "role": "user",
            "content": "补货后库存变为原来的 3 倍，现在有多少张？",
        },
        {"role": "assistant", "content": second_cot},
    ]

    dataset = QwenMultiTurnDataset(
        [{"messages": messages}],
        tokenizer,
    )
    inputs, targets = dataset[0]

    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
    )
    full_ids = encoded["input_ids"]
    if full_ids and isinstance(full_ids[0], list):
        full_ids = full_ids[0]

    from squid.data.chat_sft import find_assistant_spans

    spans = find_assistant_spans(
        tokenizer,
        messages,
        full_text,
        full_ids,
    )

    assert len(spans) == 2
    assert full_text.count("<think>") == 1

    first_expected = tokenizer.encode(
        "192<|im_end|>",
        add_special_tokens=False,
    )
    second_expected = tokenizer.encode(
        second_cot + "<|im_end|>",
        add_special_tokens=False,
    )

    (first_start, first_end), (second_start, second_end) = spans

    assert full_ids[first_start:first_end] == first_expected
    assert full_ids[second_start:second_end] == second_expected

    assert (
        targets[first_start - 1:first_end - 1]
        == first_expected
    )
    assert (
        targets[second_start - 1:second_end - 1]
        == second_expected
    )

