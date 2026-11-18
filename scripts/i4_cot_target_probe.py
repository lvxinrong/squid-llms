import json
from pathlib import Path

from transformers import AutoTokenizer

from squid.data.chat_sft import QwenMultiTurnDataset


root = Path(__file__).resolve().parents[1]
model_dir = root / "data/model_pth/qwen3/qwen3-0_6b"
tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

data = json.loads(
    (root / "data/instruct/i4_cot_synthetic_train.json")
    .read_text(encoding="utf-8")
)
dataset = QwenMultiTurnDataset(data[:2], tokenizer)

for index, (inputs, targets) in enumerate(dataset):
    supervised_ids = [
        token_id for token_id in targets
        if token_id != -100
    ]

    print(f"\n===== sample {index} =====")
    print(tokenizer.decode(supervised_ids))
    print(f"supervised tokens: {len(supervised_ids)}")

    assert tokenizer.convert_tokens_to_ids("<think>") in supervised_ids
    assert tokenizer.convert_tokens_to_ids("</think>") in supervised_ids
