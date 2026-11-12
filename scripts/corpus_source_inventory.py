"""I1 源文档盘点：在清洗前统计博客文章类型、正文规模、哈希与排除原因。

用途：先知道哪些 Markdown 能进入语料，避免草稿、索引和特殊页面被无意用于训练。
输入：lv_blog/content/**/*.md；完整清单写入 data/reports/i1-source-inventory.json。
边界：这里统计原始正文而非清洗后文本，也不执行 tokenizer；同时提供 build_corpus 复用的读取与排除函数。
"""

import hashlib
import json
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BLOG_ROOT = Path("/Users/lv/lv_repo/lv_blog")
CONTENT_DIR = BLOG_ROOT / "content"
OUTPUT_PATH = ROOT / "data/reports/i1-source-inventory.json"

DOCUMENT_TYPES = {
    "content/about.md": "profile",
    "content/roadmap.md": "roadmap",
    "content/llm-backlog.md": "backlog",
}



def read_post(path):
    """读取 Hugo 的 TOML front matter 与 Markdown 正文。"""
    raw = path.read_text(encoding="utf-8")

    if raw.startswith("+++"):
        end = raw.find("\n+++", 3)
        if end != -1:
            return tomllib.loads(raw[3:end]), raw[end + 4:]

    return {}, raw


def exclusion_reason(path, meta, body):
    if path.name.startswith("_"):
        return "index"
    if meta.get("draft") is True:
        return "draft"
    if meta.get("layout") in {"archives", "columns", "search"}:
        return "layout"
    if not body.strip():
        return "empty"
    return None


def inspect_document(path):
    raw_bytes = path.read_bytes()
    meta, body = read_post(path)
    relative_path = str(path.relative_to(BLOG_ROOT))

    return {
        "path": relative_path,
        "document_type": DOCUMENT_TYPES.get(
            relative_path,
            "article",
        ),
        "title": meta.get("title", path.stem),
        "date": str(meta.get("date", "")),
        "characters": len(body),
        "utf8_bytes": len(body.encode("utf-8")),
        "sha256": hashlib.sha256(raw_bytes).hexdigest(),
    }


def main():
    documents = []
    excluded = {}

    for path in sorted(CONTENT_DIR.rglob("*.md")):
        meta, body = read_post(path)
        reason = exclusion_reason(path, meta, body)

        if reason is not None:
            excluded[reason] = excluded.get(reason, 0) + 1
            continue

        documents.append(inspect_document(path))

    by_type = {}
    for item in documents:
        kind = item["document_type"]
        by_type[kind] = by_type.get(kind, 0) + 1

    summary = {
        "included_documents": len(documents),
        "by_type": by_type,
        "excluded": excluded,
        "characters": sum(item["characters"] for item in documents),
        "utf8_bytes": sum(item["utf8_bytes"] for item in documents),
    }

    report = {
        "source_root": str(CONTENT_DIR),
        "summary": summary,
        "documents": documents,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n源文档清单已保存到：{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
