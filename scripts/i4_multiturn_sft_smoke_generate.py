from functools import partial
from pathlib import Path

from transformers import AutoTokenizer

from squid.checkpoints.store import load_checkpoint
from squid.device import get_device
from squid.generate.chat import ChatSession, build_qwen_chat_prompt
from squid.model.v4_qwen3.gpt import GPTModelV4
from squid.tokenizer.qwen_tokenizer import QwenTokenizerAdapter
import json


def split_for_generation(messages):
    history = []

    for index in range(1, len(messages) - 2, 2):
        history.append((
            messages[index]["content"],
            messages[index + 1]["content"],
        ))

    current_user = messages[-2]["content"]
    expected = messages[-1]["content"]
    return history, current_user, expected


def main():
    root = Path(__file__).resolve().parents[1]
    model_dir = root / "data/model_pth/qwen3/qwen3-0_6b"
    checkpoint_path = (
        root / "data/model_pth/qwen3/i4-multiturn-sft-smoke.pth"
    )

    device = get_device()
    checkpoint = load_checkpoint(checkpoint_path)

    model = GPTModelV4(checkpoint["config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    tokenizer = QwenTokenizerAdapter(str(model_dir))
    prompt_builder = partial(
        build_qwen_chat_prompt,
        tokenizer.tokenizer,
        enable_thinking=False,
    )

    data = json.loads(
        (root / "data/instruct/i4_multiturn_smoke_val.json")
        .read_text(encoding="utf-8")
    )

    for entry in data:
        history, current_user, expected = split_for_generation(
            entry["messages"]
        )

        session = ChatSession(
            model,
            tokenizer,
            max_new_tokens=64,
            context_size=checkpoint["config"]["context_length"],
            prompt_builder=prompt_builder,
        )
        session.history = history

        response = session.ask(current_user)

        print("\nUser:", current_user)
        print("Expected:", expected)
        print("Generated:", response)
        print("Finish:", session.last_finish_reason)


if __name__ == "__main__":
    main()
