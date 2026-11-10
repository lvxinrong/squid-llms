import hashlib
import json
from pathlib import Path

from corpus_source_inventory import (
    BLOG_ROOT,
    CONTENT_DIR,
    DOCUMENT_TYPES,
    exclusion_reason,
    read_post,
)
from squid.data.markdown import clean_markdown


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data/corpus/blog_v1"

ALLOWED_TYPES = {"article", "profile"}

HELDOUT_PATHS = {
    "content/ai/我想把家里的设备组成一个私人AI实验室.md",
    "content/llm/手搓LLM笔记07-指令微调.md",
}


def make_document(path, meta, body):
    relative_path = str(path.relative_to(BLOG_ROOT))
    text = clean_markdown(body)

    if not text:
        return None

    return {
        "path": relative_path,
        "title": meta.get("title", path.stem),
        "document_type": DOCUMENT_TYPES.get(
            relative_path,
            "article",
        ),
        "text": text,
        "sha256": hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest(),
    }


def collect_documents():
    train_documents = []
    heldout_documents = []

    for path in sorted(CONTENT_DIR.rglob("*.md")):
        meta, body = read_post(path)

        if exclusion_reason(path, meta, body) is not None:
            continue

        relative_path = str(path.relative_to(BLOG_ROOT))
        document_type = DOCUMENT_TYPES.get(
            relative_path,
            "article",
        )

        if document_type not in ALLOWED_TYPES:
            continue

        document = make_document(path, meta, body)
        if document is None:
            continue

        if relative_path in HELDOUT_PATHS:
            heldout_documents.append(document)
        else:
            train_documents.append(document)

    return train_documents, heldout_documents


def assert_no_exact_duplicates(documents):
    seen = {}

    for document in documents:
        digest = document["sha256"]

        if digest in seen:
            raise ValueError(
                "清洗后正文完全重复：\n"
                f"- {seen[digest]}\n"
                f"- {document['path']}"
            )

        seen[digest] = document["path"]


def write_jsonl(path, documents):
    content = "".join(
        json.dumps(document, ensure_ascii=False) + "\n"
        for document in documents
    )
    path.write_text(content, encoding="utf-8")


def document_metadata(documents):
    return [
        {
            "path": document["path"],
            "title": document["title"],
            "document_type": document["document_type"],
            "characters": len(document["text"]),
            "utf8_bytes": len(document["text"].encode("utf-8")),
            "sha256": document["sha256"],
        }
        for document in documents
    ]


def main():
    train_documents, heldout_documents = collect_documents()

    assert heldout_documents, "没有找到固定 held-out 文档"
    assert len(heldout_documents) == len(HELDOUT_PATHS)

    assert_no_exact_duplicates(
        train_documents + heldout_documents
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    write_jsonl(
        OUTPUT_DIR / "train.jsonl",
        train_documents,
    )
    write_jsonl(
        OUTPUT_DIR / "heldout.jsonl",
        heldout_documents,
    )

    manifest = {
        "version": "blog_v1",
        "source_root": str(CONTENT_DIR),
        "cleaner": "squid.data.markdown.clean_markdown",
        "summary": {
            "train_documents": len(train_documents),
            "heldout_documents": len(heldout_documents),
            "total_documents": (
                len(train_documents) + len(heldout_documents)
            ),
        },
        "train": document_metadata(train_documents),
        "heldout": document_metadata(heldout_documents),
    }

    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(manifest["summary"], indent=2))
    print(f"\n语料已保存到：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
