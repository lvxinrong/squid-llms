import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from squid.checkpoints.qwen3 import load_qwen3_dir
from squid.data.chat_sft import build_assistant_targets
from squid.device import get_device
import math



def encode_response(tokenizer, messages, response):
    prefix_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    full_text = prefix_text + response + "<|im_end|>\n"

    prefix_ids = tokenizer.encode(
        prefix_text,
        add_special_tokens=False,
    )
    full_ids = tokenizer.encode(
        full_text,
        add_special_tokens=False,
    )

    start = len(prefix_ids)
    end = len(full_ids) - 1

    assert prefix_ids == full_ids[:start]

    inputs = full_ids[:-1]
    targets = build_assistant_targets(
        full_ids,
        assistant_spans=[(start, end)],
    )
    return inputs, targets


def sequence_logprob(
    model,
    input_ids,
    target_ids,
    device,
    require_grad=False,
):
    inputs = torch.tensor(
        [input_ids],
        dtype=torch.long,
        device=device,
    )
    targets = torch.tensor(
        [target_ids],
        dtype=torch.long,
        device=device,
    )

    with torch.set_grad_enabled(require_grad):
        logits = model(inputs)
        log_probs = torch.log_softmax(logits, dim=-1)

        safe_targets = targets.clone()
        safe_targets[safe_targets == -100] = 0

        token_logprobs = log_probs.gather(
            -1,
            safe_targets.unsqueeze(-1),
        ).squeeze(-1)

        mask = targets != -100
        total = (token_logprobs * mask).sum()

    return total, int(mask.sum())



def main():
    root = Path(__file__).resolve().parents[1]
    model_dir = root / "data/model_pth/qwen3/qwen3-0_6b"
    device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = load_qwen3_dir(
        model_dir,
        context_size=1024,
        device=device,
    )

    data = json.loads(
        (root / "data/instruct/i4_dpo_preference_probe.json")
        .read_text(encoding="utf-8")
    )
    entry = data[0]

    results = {}

    for key in ("chosen", "rejected"):
        input_ids, target_ids = encode_response(
            tokenizer,
            entry["messages"],
            entry[key],
        )
        total, token_count = sequence_logprob(
            model,
            input_ids,
            target_ids,
            device,
        )

        results[key] = {
            "sequence_logprob": total.item(),
            "token_count": token_count,
            "mean_logprob_per_token": (
                total.item() / token_count
            ),
        }

    results["chosen_minus_rejected"] = (
        results["chosen"]["sequence_logprob"]
        - results["rejected"]["sequence_logprob"]
    )

    beta = 0.1

    reference_chosen = results["chosen"]["sequence_logprob"]
    reference_rejected = results["rejected"]["sequence_logprob"]
    policy_chosen = reference_chosen
    policy_rejected = reference_rejected

    score = (
        (policy_chosen - reference_chosen)
        - (policy_rejected - reference_rejected)
    )
    loss = -torch.nn.functional.logsigmoid(
        torch.tensor(beta * score)
    )

    results["dpo"] = {
        "beta": beta,
        "score": score,
        "loss": loss.item(),
        "ln_2": math.log(2),
    }

    assert math.isclose(score, 0.0, abs_tol=1e-9)
    assert math.isclose(loss.item(), math.log(2), rel_tol=1e-6)

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
