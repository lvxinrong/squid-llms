"""I3 对话服务：提供同步 /chat、SSE /chat/stream 与有界多会话隔离。

用途：在同一本地 Qwen 模型上支持逐 token 推送和按 session_id 隔离的 history / cache。
输入：本地 V4 / Qwen3-0.6B  权重与 tokenizer；默认监听 127.0.0.1:8000。
边界：HTTPServer 仍串行处理请求，不代表支持并发模型推理；会话使用有界 LRU，断开的流会失效对应 cache。
"""

import json
import os
from functools import partial
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from squid.checkpoints.qwen3 import load_qwen3_dir
from squid.device import get_device
from squid.generate.chat import ChatSession, build_qwen_chat_prompt
from squid.generate.session_store import SessionStore
from squid.tokenizer.qwen_tokenizer import QwenTokenizerAdapter


SESSION_STORE = None


class ChatHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlsplit(self.path)

        if parsed.path == "/":
            root = Path(__file__).resolve().parents[1]
            body = (root / "web/chat.html").read_bytes()
            content_type = "text/html; charset=utf-8"
        elif parsed.path == "/health":
            body = json.dumps({
                "ok": True,
                "sessions": len(SESSION_STORE),
            }).encode("utf-8")
            content_type = "application/json"
        elif parsed.path == "/sessions":
            body = json.dumps({
                "sessions": SESSION_STORE.list_sessions(),
            }, ensure_ascii=False).encode("utf-8")
            content_type = "application/json; charset=utf-8"
        elif parsed.path == "/session/history":
            session_id = parse_qs(parsed.query).get("session_id", [""])[0]
            if not session_id:
                self.send_response(400)
                self.end_headers()
                return

            session = SESSION_STORE.find(session_id)
            history = [] if session is None else [
                {"user": user, "assistant": assistant}
                for user, assistant in session.history
            ]
            body = json.dumps({
                "session_id": session_id,
                "history": history,
                "turns": len(history),
            }, ensure_ascii=False).encode("utf-8")
            content_type = "application/json; charset=utf-8"
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
        if self.path not in {"/chat", "/chat/stream"}:
            self.send_response(404)
            self.end_headers()
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
        except (ValueError, json.JSONDecodeError):
            self.send_response(400)
            self.end_headers()
            return

        message = payload.get("message", "").strip()
        if not message:
            self.send_response(400)
            self.end_headers()
            return

        session_id = payload.get("session_id") or "default"
        try:
            session = SESSION_STORE.get(session_id)
        except ValueError:
            self.send_response(400)
            self.end_headers()
            return

        enable_thinking = payload.get("enable_thinking", True)
        if not isinstance(enable_thinking, bool):
            self.send_response(400)
            self.end_headers()
            return

        session.prompt_builder = partial(
            build_qwen_chat_prompt,
            session.tokenizer.tokenizer,
            enable_thinking=enable_thinking,
        )

        if self.path == "/chat/stream":
            self.stream_chat(session, session_id, message)
            return

        response = session.ask(message)
        body = json.dumps(
            {"response": response, "session_id": session_id},
            ensure_ascii=False,
        ).encode("utf-8")

        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def write_sse(self, payload):
        body = json.dumps(payload, ensure_ascii=False)
        self.wfile.write(f"data: {body}\n\n".encode("utf-8"))
        self.wfile.flush()

    def stream_chat(self, session, session_id, message):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        try:
            for delta in session.ask_stream(message):
                if delta:
                    self.write_sse({
                        "session_id": session_id,
                        "delta": delta,
                    })
            self.write_sse({
                "session_id": session_id,
                "done": True,
                "finish_reason": session.last_finish_reason,
                "generated_tokens": session.last_generated_tokens,
                "max_new_tokens": session.max_new_tokens,
            })
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as error:
            self.write_sse({
                "session_id": session_id,
                "done": True,
                "error": str(error),
            })
        finally:
            self.close_connection = True


def main():
    global SESSION_STORE

    root = Path(__file__).resolve().parents[1]
    # weights_path = root / "data/model_pth/qwen/qwen2_5-0_5b-instruct.safetensors"
    model_dir = root / "data/model_pth/qwen3/qwen3-0_6b"
    context_size = int(os.getenv("LVCHAT_CONTEXT_SIZE", "8192"))
    max_new_tokens = int(os.getenv("LVCHAT_MAX_NEW_TOKENS", "1024"))
    max_sessions = int(os.getenv("LVCHAT_MAX_SESSIONS", "16"))

    # model = load_qwen2_5(
    #     weights_path,
    #     cfg={"context_length": context_size},
    #     device=get_device(),
    # )
    # tokenizer = QwenTokenizerAdapter("Qwen/Qwen2.5-0.5B-Instruct")
    model = load_qwen3_dir(
        model_dir,
        context_size=context_size,
        device=get_device(),
    )
    tokenizer = QwenTokenizerAdapter(str(model_dir))
    prompt_builder = partial(
        build_qwen_chat_prompt,
        tokenizer.tokenizer,
    )

    def session_factory():
        return ChatSession(
            model,
            tokenizer,
            max_new_tokens=max_new_tokens,
            context_size=context_size,
            prompt_builder=prompt_builder,
        )

    SESSION_STORE = SessionStore(
        session_factory,
        max_sessions=max_sessions,
    )

    server = HTTPServer(("127.0.0.1", 8000), ChatHandler)
    print("lvchat server: http://127.0.0.1:8000/")
    print(f"SSE endpoint: POST /chat/stream · max sessions: {max_sessions}")
    server.serve_forever()


if __name__ == "__main__":
    main()
