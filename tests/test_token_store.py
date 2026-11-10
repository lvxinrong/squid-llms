import json
from pathlib import Path

import numpy as np

from squid.data.token_stream import pack_tokens
from squid.tokenizer.qwen_tokenizer import QwenTokenizerAdapter

ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = ROOT / "data/corpus/blog_v1"
STORE_DIR = ROOT / "data/token_store/blog_v1"

def read_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def test_token_store_matches_pack_tokens():
    metadata = json.loads(
        (STORE_DIR / "metadata.json").read_text(encoding="utf-8")
    )
    tokenizer = QwenTokenizerAdapter(metadata["tokenizer"])

    for split_name in ("train", "heldout"):
        split = metadata["splits"][split_name]
        documents = read_jsonl(
            CORPUS_DIR / f"{split_name}.jsonl"
        )

        texts = [
            f"# {document['title']}\n\n{document['text']}"
            for document in documents
        ]

        expected = pack_tokens(
            texts,
            tokenizer,
            metadata["eos_id"],
        )

        actual = np.memmap(
            STORE_DIR / split["file"],
            dtype=metadata["dtype"],
            mode="r",
        )

        assert len(actual) == split["token_count"]
        assert len(actual) == len(expected)

        np.testing.assert_array_equal(
            actual,
            np.asarray(expected, dtype=np.uint32),
        )

        expected_offset = 0

        for document, document_meta in zip(
                documents,
                split["documents"],
        ):
            assert document_meta["path"] == document["path"]
            assert document_meta["offset"] == expected_offset

            encoded_length = len(
                tokenizer.encode(
                    f"# {document['title']}\n\n{document['text']}"
                )
            ) + 1

            assert document_meta["length"] == encoded_length
            assert actual[expected_offset + encoded_length - 1] == (
                metadata["eos_id"]
            )

            expected_offset += encoded_length

        assert expected_offset == len(expected)