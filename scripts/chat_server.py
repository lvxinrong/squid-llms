import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from squid.device import get_device
from squid.generate.chat import ChatSession, build_qwen_chat_prompt
from squid.checkpoints.qwen import load_qwen2_5
from squid.tokenizer.qwen_tokenizer import QwenTokenizerAdapter
from functools import partial

SESSION = None


class ChatHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            root = Path(__file__).resolve().parents[1]
            body = (root / "web/chat.html").read_bytes()
            content_type = "text/html; charset=utf-8"
        elif self.path == "/health":
            body = b'{"ok":true}'
            content_type = "application/json"
        else:
            self.send_response(404)
            self.end_headers()
            return

        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/chat":
            self.send_response(404)
            self.end_headers()
            return

        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length))
        message = payload.get("message", "").strip()

        if not message:
            self.send_response(400)
            self.end_headers()
            return

        response = SESSION.ask(message)
        body = json.dumps(
            {"response": response},
            ensure_ascii=False,
        ).encode("utf-8")

        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)



def main():
    global SESSION
    root = Path(__file__).resolve().parents[1]
    weights_path = root / "data/model_pth/qwen/qwen2_5-0_5b-instruct.safetensors"

    model = load_qwen2_5(weights_path, device=get_device())
    tokenizer = QwenTokenizerAdapter("Qwen/Qwen2.5-0.5B-Instruct")

    prompt_builder = partial(
        build_qwen_chat_prompt,
        tokenizer.tokenizer,
    )

    SESSION = ChatSession(
        model,
        tokenizer,
        max_new_tokens=128,
        context_size=1024,
        prompt_builder=prompt_builder,
    )

    server = HTTPServer(("127.0.0.1", 8000), ChatHandler)
    print("lvchat server: http://127.0.0.1:8000/")
    server.serve_forever()


if __name__ == "__main__":
    main()
