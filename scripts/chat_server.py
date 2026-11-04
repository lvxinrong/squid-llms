import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from squid.checkpoints.store import load_model
from squid.device import get_device
from squid.generate.chat import ChatSession
from squid.tokenizer import TiktokenTokenizer

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
    checkpoint_path = root / "data/model_pth/gpt2_355M_instruct.pth"

    model, _ = load_model(checkpoint_path, get_device())
    model.eval()

    tokenizer = TiktokenTokenizer("gpt2")
    SESSION = ChatSession(model, tokenizer)

    server = HTTPServer(("127.0.0.1", 8000), ChatHandler)
    print("lvchat server: http://127.0.0.1:8000/chat")
    server.serve_forever()


if __name__ == "__main__":
    main()
