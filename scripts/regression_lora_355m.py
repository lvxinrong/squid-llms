"""M0 回归：旧仓库 LoRA 权重（23.9 分基线）→ squid 加载 → 生成 → 打分。

用法：
    uv run python scripts/regression_lora_355m.py --limit 2 --no-score   # 冒烟
    uv run python scripts/regression_lora_355m.py                        # 全量
"""
import argparse
import json
from pathlib import Path

from squid.checkpoints.store import load_checkpoint
from squid.device import get_device
from squid.eval.generate import generate_responses
from squid.eval.judge import score_entries, summarize_scores
from squid.model.gpt import GPTModel
from squid.model.lora import replace_linear_with_lora
from squid.tokenizer import TiktokenTokenizer

OLD_REPO = Path("/Users/lv/lv_repo/lv_llms")
WEIGHTS = OLD_REPO / "gpt2_355M_instruct.pth"
OUT = Path("regression")
RANK, ALPHA = 16, 16


def load_old_lora_model(device):
    ckpt = load_checkpoint(WEIGHTS)
    model = GPTModel(ckpt["config"])
    replace_linear_with_lora(model, rank=RANK, alpha=ALPHA)
    state_dict = {
        k.replace("trf_blocks.", "trf_block.").replace(".ff.", ".ffn."): v
        for k, v in ckpt["model_state_dict"].items()
    }
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model, ckpt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 条（冒烟用）")
    parser.add_argument("--no-score", action="store_true", help="只生成，不打分")
    args = parser.parse_args()

    OUT.mkdir(exist_ok=True)
    device = get_device()
    model, ckpt = load_old_lora_model(device)
    tokenizer = TiktokenTokenizer("gpt2")

    data = json.loads((OLD_REPO / "instruction-data.json").read_text(encoding="utf-8"))
    test_data = data[990:]           # 书的划分：train 935 / val 55 / test 110
    assert len(test_data) == 110
    if args.limit:
        test_data = test_data[:args.limit]

    responses = generate_responses(
        model, tokenizer, test_data, OUT / "responses.json",
        max_new_tokens=256, context_size=ckpt["config"]["context_length"],
    )
    print(f"生成完成 {len(responses)} 条")
    if args.no_score:
        print("样例 response:", responses[0]["model_response"][:200])
        return

    scored = score_entries(responses, save_path=OUT / "scores.json")
    print("\n=== squid 回归结果 ===")
    summarize_scores(scored)
    print("\n=== 旧仓库基线（同权重，23.9 分那次） ===")
    old = json.loads((OLD_REPO / "instruction-data-with-scores.json").read_text(encoding="utf-8"))
    summarize_scores(old)


if __name__ == "__main__":
    main()
