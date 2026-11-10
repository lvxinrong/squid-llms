import hashlib
import json
import time
from pathlib import Path

from squid.tokenizer.qwen_tokenizer import QwenTokenizerAdapter

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/blog"
OUTPUT_PATH = ROOT / "data/reports/i1-corpus-manifest.json"
TOKENIZER_NAME = "Qwen/Qwen2.5-0.5B"

def inspect_document(path, tokenizer):
    """
    raw：文件的原始字节内容，类型是 bytes。
    path.read_bytes() 会把整个文件按二进制读出来，不做任何编码解释。比如文件里的中文、换行符、BOM 等在 raw 里都是字节。

    text：把 raw 按 UTF-8 编码解码后的文本字符串，类型是 str。
    raw.decode("utf-8") 表示“这些字节是用 UTF-8 编码的，请把它转成 Python 字符串”。
    """
    raw = path.read_bytes()
    text = raw.decode("utf-8")

    start = time.perf_counter()
    token_ids = tokenizer.encode(text)
    encode_seconds = time.perf_counter() - start

    return {
        "path": str(path.relative_to(ROOT)),
        "split": "train" if path.name == "train.txt" else "heldout",
        "characters": len(text),
        "utf8_bytes": len(raw),
        "tokens": len(token_ids),
        "encode_seconds": encode_seconds,
        "sha256": hashlib.sha256(raw).hexdigest(),
    }

def main():
    tokenizer = QwenTokenizerAdapter(TOKENIZER_NAME)

    documents = [
        inspect_document(path, tokenizer) for path in sorted(DATA_DIR.glob("*.txt"))
    ]

    summary = {
        "documents": len(documents),
        "characters": sum(item["characters"] for item in documents),
        "utf8_bytes": sum(item["utf8_bytes"] for item in documents),
        "tokens": sum(item["tokens"] for item in documents),
        "encode_seconds": sum(
            item["encode_seconds"] for item in documents
        ),
    }

    manifest = {
        "tokenizer": TOKENIZER_NAME,
        "documents": documents,
        "summary": summary,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nmanifest 已保存到：{OUTPUT_PATH}")

if __name__ == '__main__':
    main()

