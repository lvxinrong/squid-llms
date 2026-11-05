import json
from pathlib import Path

from squid.checkpoints.store import load_model
from squid.device import get_device
from squid.eval.arithmetic import evaluate_arithmetic
from squid.tokenizer import TiktokenTokenizer


def test_gpt_edited_hard_score():
    root = Path("/Users/lv/lv_repo/squid-llms")
    checkpoint_path = root / "data/model_pth/experiments/weight_edit/gpt2_355M_instruct_gpt_v1.pth"
    model, checkpoint = load_model(checkpoint_path, get_device())
    model.eval()

    tokenizer = TiktokenTokenizer("gpt2")
    test_data_path = root / "data/arithmetic/test_data_hard.json"
    test_data = json.loads(test_data_path.read_text(encoding="utf-8"))

    report = evaluate_arithmetic(
        model,
        tokenizer,
        test_data,
        root / "data/arithmetic/gpt-edited-hard-report.json",
        context_size=checkpoint["config"]["context_length"],
        description="GPT edited arithmetic hard set",
    )

    print("\n=== GPT 权重编辑后的中等难度四则运算验收 ===")
    print(json.dumps(report["summary"], indent=4, ensure_ascii=False))
    assert len(report["items"]) == 200
