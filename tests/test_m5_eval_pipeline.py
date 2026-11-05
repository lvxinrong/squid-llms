import json
from pathlib import Path
from unittest.mock import patch

from squid.checkpoints.store import load_model
from squid.device import get_device
from squid.eval.generate import generate_response_resumable
from squid.eval.judge import build_score_summary, save_score_report, score_entries
from squid.tokenizer import TiktokenTokenizer


def test_score_summary_and_report(tmp_path):
    data = [
        {"score": 0},
        {"score": 20},
        {"score": 40},
        {"score": 60},
        {"score": 80},
        {"score": 100},
        {"score": None},
    ]

    summary = build_score_summary(data)
    assert summary["total"] == 7
    assert summary["valid_scores"] == 6
    assert summary["mean"] == 50
    assert summary["median"] == 50
    assert summary["min"] == 0
    assert summary["max"] == 100
    assert summary["distribution"] == {
        "0": 1,
        "1-20": 1,
        "21-40": 1,
        "41-60": 1,
        "61-80": 1,
        "81-100": 1,
    }

    report_path = tmp_path / "score-report.json"
    report = save_score_report(data, report_path)
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved == report
    assert saved["items"] == data


def test_m5_sft_eval_pipeline():
    root = Path("/Users/lv/lv_repo/squid-llms")
    tokenizer = TiktokenTokenizer("gpt2")
    device = get_device()
    model, _ = load_model(root / "data/model_pth/gpt2/gpt2_355M_instruct.pth", device)
    model.eval()

    data = json.loads(
        (root / "data/instruct/instruction-data.json").read_text(encoding="utf-8")
    )
    train_portion = int(len(data) * 0.85)
    test_portion = int(len(data) * 0.1)
    test_data = data[train_portion:train_portion + test_portion]

    response_path = root / "data/instruct/m5-response-sft-110.json"
    scored_path = root / "data/instruct/m5-scored-sft-110.json"
    report_path = root / "data/instruct/m5-report-sft-110.json"

    responses = generate_response_resumable(
        model,
        tokenizer,
        test_data,
        response_path,
    )
    assert len(responses) == 110
    assert all(
        "<|endoftext|>" not in item["model_response"]
        for item in responses
    )

    with patch(
        "squid.eval.generate.generate_text",
        side_effect=AssertionError("已生成记录不应再次调用模型"),
    ):
        resumed = generate_response_resumable(
            model,
            tokenizer,
            test_data,
            response_path,
        )
    assert resumed == responses

    scored = score_entries(
        json.loads(response_path.read_text(encoding="utf-8")),
        save_path=str(scored_path),
        flush_every=1,
    )
    report = save_score_report(scored, report_path)

    print("\n=== M5 SFT 110 条评估报告 ===")
    print(json.dumps(report["summary"], indent=4, ensure_ascii=False))
    assert report["summary"]["total"] == 110
    assert report["summary"]["valid_scores"] == 110
