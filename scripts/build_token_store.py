import hashlib
import json
import numpy as np
import sys
from pathlib import Path
from squid.tokenizer.qwen_tokenizer import QwenTokenizerAdapter

ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = ROOT / "data/corpus/blog_v1"
OUTPUT_DIR = ROOT / "data/token_store/blog_v1"
TOKENIZER_NAME = "Qwen/Qwen2.5-0.5B"

def write_tokens(jsonl_path, output_path, tokenizer):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_tokens = 0
    documents = []

    with (
        jsonl_path.open("r", encoding="utf-8") as source,
        output_path.open("wb") as target,
    ):
        for line in source:
            # 判断这一行是否为空行或只包含空白字符，如果是就跳过
            if not line.strip():
                continue

            document = json.loads(line)
            text = (
                f"# {document['title']}\n\n"
                f"{document['text']}"
            )
            encoded_sha256 = hashlib.sha256(
                text.encode("utf-8")
            ).hexdigest()

            ids = tokenizer.encode(text)
            ids.append(tokenizer.eos_id)

            offset = total_tokens
            token_array = np.asarray(ids, dtype=np.uint32)
            token_array.tofile(target)

            total_tokens += len(ids)
            documents.append({
                "path": document["path"],
                "title": document["title"],
                "offset": offset,
                "length": len(ids),
                "source_sha256": document["sha256"],
                "encoded_sha256": encoded_sha256,
            })

            print(
                f"{document['path']}: "
                f"{len(ids)} tokens（含 EOS）"
            )

    return {
        "token_count": total_tokens,
        "documents": documents,
    }

def make_split_metadata(result, filename):
    file_path = OUTPUT_DIR /filename
    expected_bytes = result["token_count"] * 4
    actual_bytes = file_path.stat().st_size

    assert actual_bytes == expected_bytes, (
        f"{filename} 文件大小异常："
        f"期望 {expected_bytes}，实际 {actual_bytes}"
    )

    return {
        "file": filename,
        "file_bytes": actual_bytes,
        "token_count": result["token_count"],
        "document_count": len(result["documents"]),
        "documents": result["documents"],
    }

def main():
    tokenizer = QwenTokenizerAdapter(TOKENIZER_NAME)

    train_result = write_tokens(
        CORPUS_DIR / "train.jsonl",
        OUTPUT_DIR / "train.uint32.bin",
        tokenizer,
    )

    heldout_result = write_tokens(
        CORPUS_DIR / "heldout.jsonl",
        OUTPUT_DIR / "heldout.uint32.bin",
        tokenizer,
    )

    metadata = {
        "version": "blog_v1",
        "tokenizer": TOKENIZER_NAME,
        "eos_id": tokenizer.eos_id,
        "dtype": "uint32",
        "bytes_per_token": 4,
        "byte_order": sys.byteorder,
        "include_title": True,
        "title_template": "# {title}\n\n{text}",
        "corpus_manifest": "data/corpus/blog_v1/manifest.json",
        "splits": {
            "train": make_split_metadata(
                train_result,
                "train.uint32.bin",
            ),
            "heldout": make_split_metadata(
                heldout_result,
                "heldout.uint32.bin",
            ),
        },
    }

    metadata_path = OUTPUT_DIR / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n===== Token Store 汇总 =====")
    print(
        json.dumps(
            {
                "train_tokens": train_result["token_count"],
                "heldout_tokens": heldout_result["token_count"],
                "metadata": str(metadata_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()



