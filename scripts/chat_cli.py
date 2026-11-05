from pathlib import Path

from squid.checkpoints.store import load_model
from squid.device import get_device
from squid.generate.chat import ChatSession
from squid.tokenizer import TiktokenTokenizer


def main():
    root = Path(__file__).resolve().parents[1]
    checkpoint_path = root / "data/model_pth/gpt2/gpt2_355M_instruct.pth"
    model, _ = load_model(checkpoint_path, get_device())
    model.eval()
    tokenizer = TiktokenTokenizer("gpt2")
    session = ChatSession(model, tokenizer)
    print("lvchat CLI 已启动。输入 /clear 清空历史，/exit 退出。")

    while True:
        user_input = input("\n用户:").strip()
        if not user_input:
            continue

        if user_input == "/exit":
            print("已退出 lvchat")
            break

        if user_input == "/clear":
            session.clear()
            print("历史对话已经清空")
            continue

        response = session.ask(user_input)
        print(f"模型：{response}")
if __name__ == "__main__":
    main()