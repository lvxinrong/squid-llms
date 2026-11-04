import json
from pathlib import Path

from squid.checkpoints.store import load_model
from squid.device import get_device
from squid.eval.arithmetic import evaluate_arithmetic
from squid.tokenizer import TiktokenTokenizer


def test_base_line_score():
    tokenizer = TiktokenTokenizer("gpt2")
    root = Path("/Users/lv/lv_repo/squid-llms")
    sft_path = root / "data/model_pth/gpt2_355M_instruct.pth"
    model, checkpoint = load_model(sft_path, get_device())
    model.eval()

    test_data_path = root / "data/arithmetic/test_data.json"
    test_data = json.loads(test_data_path.read_text(encoding="utf-8"))
    report = evaluate_arithmetic(
        model,
        tokenizer,
        test_data,
        root / "data/arithmetic/sft-baseline-report.json",
        context_size=checkpoint["config"]["context_length"],
        description="SFT arithmetic baseline",
    )

    print("\n=== 四则运算基线 ===")
    print(json.dumps(report["summary"], indent=4, ensure_ascii=False))
    assert len(report["items"]) == 200
