import json
from functools import partial
from pathlib import Path

from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from squid.data.chat_sft import (
    QwenMultiTurnDataset,
    qwen_multiturn_collate,
)


root = Path(__file__).resolve().parents[1]
model_dir = root / "data/model_pth/qwen3/qwen3-0_6b"
tokenizer = AutoTokenizer.from_pretrained(str(model_dir))


def audit_split(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    dataset = QwenMultiTurnDataset(data, tokenizer)

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=partial(
            qwen_multiturn_collate,
            pad_token_id=tokenizer.eos_token_id,
        ),
    )

    print(f"\n{path.name}: {len(dataset)} samples")

    for inputs, targets in loader:
        supervised = int((targets != -100).sum())

        print(
            f"batch={tuple(inputs.shape)}, "
            f"supervised_tokens={supervised}"
        )

        assert inputs.shape == targets.shape
        assert supervised > 0


audit_split(root / "data/instruct/i4_multiturn_smoke_train.json")
audit_split(root / "data/instruct/i4_multiturn_smoke_val.json")
