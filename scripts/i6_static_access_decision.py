import hashlib
import json
import runpy
from pathlib import Path

import torch
from transformers import AutoTokenizer

from squid.checkpoints.lora import load_lora
from squid.checkpoints.qwen3 import load_qwen3_dir
from squid.device import get_device
from squid.generate.engine import generate


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "data/model_pth/qwen3/qwen3-0_6b"
DATA_PATH = (
    ROOT / "data/instruct/i6_static_access_dev_pairs_v1.json"
)
ADAPTER_PATH = (
    ROOT / "data/model_pth/qwen3/i6-l0-static-v2-lora.pth"
)
REPORT_PATH = (
    ROOT / "data/reports/i6-static-access-dev-decision.json"
)
CONTEXT_SIZE = 4096
MAX_NEW_TOKENS = 16
LABELS = ("非法", "合法")


def build_prompt(tokenizer, question):
    return tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": question,
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def generate_answer(model, tokenizer, prompt, device):
    prompt_ids = tokenizer.encode(
        prompt,
        add_special_tokens=False,
    )
    input_ids = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=device,
    )

    with torch.no_grad():
        generated = generate(
            model,
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            context_size=CONTEXT_SIZE,
            temperature=0.0,
            eos_id=tokenizer.eos_token_id,
        )

    generated_ids = generated[0, len(prompt_ids):].tolist()
    stopped = bool(
        generated_ids
        and generated_ids[-1] == tokenizer.eos_token_id
    )
    content_ids = (
        generated_ids[:-1]
        if stopped
        else generated_ids
    )
    response = tokenizer.decode(
        content_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ).strip()

    return {
        "response": response,
        "finish_reason": "stop" if stopped else "length",
        "generated_tokens": len(generated_ids),
    }


def score_candidates(model, tokenizer, prompt, device):
    prompt_ids = tokenizer.encode(
        prompt,
        add_special_tokens=False,
    )
    scores = {}

    for label in LABELS:
        answer_ids = tokenizer.encode(
            label,
            add_special_tokens=False,
        ) + [tokenizer.eos_token_id]

        full_ids = prompt_ids + answer_ids
        input_ids = torch.tensor(
            [full_ids[:-1]],
            dtype=torch.long,
            device=device,
        )

        with torch.no_grad():
            logits = model(input_ids)

        candidate_logits = logits[
            0,
            len(prompt_ids) - 1:,
            :,
        ].float()
        targets = torch.tensor(
            answer_ids,
            dtype=torch.long,
            device=device,
        )

        assert candidate_logits.shape[0] == len(answer_ids)

        log_probs = torch.log_softmax(
            candidate_logits,
            dim=-1,
        )
        token_log_probs = log_probs.gather(
            1,
            targets.unsqueeze(1),
        ).squeeze(1)

        scores[label] = {
            "token_ids": answer_ids,
            "token_count": len(answer_ids),
            "sum_logprob": float(token_log_probs.sum().item()),
            "mean_logprob": float(token_log_probs.mean().item()),
        }

    prediction = max(
        LABELS,
        key=lambda label: scores[label]["mean_logprob"],
    )

    return scores, prediction


def evaluate_model(model, tokenizer, payload, device, model_name):
    pair_results = []

    for pair in payload["pairs"]:
        pair_result = {
            "pair_id": pair["pair_id"],
            "cases": {},
        }

        for expected in ("illegal", "legal"):
            question = payload["question_template"].format(
                source=pair[f"{expected}_source"]
            )
            prompt = build_prompt(tokenizer, question)
            generated = generate_answer(
                model,
                tokenizer,
                prompt,
                device,
            )
            scores, score_prediction = score_candidates(
                model,
                tokenizer,
                prompt,
                device,
            )
            expected_label = payload["labels"][expected]

            case = {
                "expected": expected_label,
                "question": question,
                "response": generated["response"],
                "finish_reason": generated["finish_reason"],
                "generation_correct": (
                    generated["response"] == expected_label
                ),
                "score_prediction": score_prediction,
                "score_correct": score_prediction == expected_label,
                "candidate_scores": scores,
                "score_margin_legal_minus_illegal": (
                    scores["合法"]["mean_logprob"]
                    - scores["非法"]["mean_logprob"]
                ),
            }
            pair_result["cases"][expected] = case

            print(
                model_name,
                pair["pair_id"],
                expected,
                f"response={case['response']!r}",
                f"score={score_prediction}",
                f"margin={case['score_margin_legal_minus_illegal']:+.6f}",
            )

        illegal = pair_result["cases"]["illegal"]
        legal = pair_result["cases"]["legal"]

        pair_result["generation_pair_correct"] = (
            illegal["generation_correct"]
            and legal["generation_correct"]
        )
        pair_result["score_pair_correct"] = (
            illegal["score_correct"]
            and legal["score_correct"]
        )
        pair_result["generation_same_answer"] = (
            illegal["response"] == legal["response"]
        )
        pair_result["score_same_answer"] = (
            illegal["score_prediction"]
            == legal["score_prediction"]
        )

        pair_results.append(pair_result)
        print(
            model_name,
            pair["pair_id"],
            f"generation_pair={pair_result['generation_pair_correct']}",
            f"score_pair={pair_result['score_pair_correct']}",
        )

    pair_count = len(pair_results)
    summary = {
        "pair_count": pair_count,
        "generation_correct_pairs": sum(
            pair["generation_pair_correct"]
            for pair in pair_results
        ),
        "score_correct_pairs": sum(
            pair["score_pair_correct"]
            for pair in pair_results
        ),
        "generation_same_answer_pairs": sum(
            pair["generation_same_answer"]
            for pair in pair_results
        ),
        "score_same_answer_pairs": sum(
            pair["score_same_answer"]
            for pair in pair_results
        ),
    }

    print(model_name, "summary:", json.dumps(
        summary,
        ensure_ascii=False,
    ))

    return {
        "summary": summary,
        "pairs": pair_results,
    }


def main():
    data_text = DATA_PATH.read_text(encoding="utf-8")
    payload = json.loads(data_text)
    assert payload["data_role"] == "development"
    assert len(payload["pairs"]) == 4

    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = load_qwen3_dir(
        MODEL_DIR,
        context_size=CONTEXT_SIZE,
        device=device,
        dtype=torch.float32,
    )
    model.eval()

    report = {
        "dataset": str(DATA_PATH),
        "dataset_sha256": hashlib.sha256(
            data_text.encode("utf-8")
        ).hexdigest(),
        "dtype": "torch.float32",
        "generation": "greedy",
        "candidate_score": "mean logprob of label and EOS",
        "models": {},
    }

    report["models"]["base"] = evaluate_model(
        model,
        tokenizer,
        payload,
        device,
        "base",
    )

    namespace = runpy.run_path(str(
        ROOT / "scripts/i6_structured_edit_baseline.py"
    ))
    rank, alpha = namespace["read_adapter_config"](
        ADAPTER_PATH
    )
    namespace["add_lora"](model, rank=rank, alpha=alpha)
    load_lora(model, ADAPTER_PATH, strict=True)
    model.to(device=device, dtype=torch.float32)
    model.eval()

    report["models"]["static_v2_lr1e-5"] = evaluate_model(
        model,
        tokenizer,
        payload,
        device,
        "static_v2_lr1e-5",
    )
    report["adapter"] = str(ADAPTER_PATH)
    report["adapter_rank"] = rank
    report["adapter_alpha"] = alpha

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("report:", REPORT_PATH)


if __name__ == "__main__":
    main()
