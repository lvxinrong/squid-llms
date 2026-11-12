"""I1 token store 验收：用 memmap 只读检查 token 数、文件字节和首篇文档边界。

用途：确认 metadata 中的 offset / length / EOS 能准确定位并解码原始文档，而不是只检查文件存在。
输入：data/token_store/blog_v1/；输出终端打印首篇标题、位置和前 200 字符，不写报告。
边界：这是结构完整性检查，不评估模型训练质量；依赖 metadata 指定的 Qwen tokenizer。
"""

import json
from pathlib import Path

import numpy as np

from squid.tokenizer.qwen_tokenizer import QwenTokenizerAdapter


ROOT = Path(__file__).resolve().parents[1]
STORE_DIR = ROOT / "data/token_store/blog_v1"


def main():
    metadata = json.loads(
        (STORE_DIR / "metadata.json").read_text(encoding="utf-8")
    )
    train = metadata["splits"]["train"]

    token_path = STORE_DIR / train["file"]
    # np.memmap 的核心作用是将磁盘上的一个大文件“伪装”成一个 NumPy 数组。它允许你像操作内存中的数组一样，去读取和修改磁盘上远超系统内存大小的文件，而无需一次性将整个文件加载到 RAM 中。
    tokens = np.memmap(
        token_path,
        dtype=metadata["dtype"],
        mode="r",
    )

    assert len(tokens) == train["token_count"]
    assert token_path.stat().st_size == train["file_bytes"]

    first_document = train["documents"][0]
    start = first_document["offset"]
    end = start + first_document["length"]

    document_tokens = tokens[start:end]

    assert len(document_tokens) == first_document["length"]
    assert document_tokens[-1] == metadata["eos_id"]

    tokenizer = QwenTokenizerAdapter(metadata["tokenizer"])
    text = tokenizer.decode(document_tokens[:-1].tolist())

    print("memmap shape:", tokens.shape)
    print("dtype:", tokens.dtype)
    print("mode:", tokens.mode)
    print("首篇标题:", first_document["title"])
    print("offset:", first_document["offset"])
    print("length:", first_document["length"])
    print("\n===== 首篇前 200 字符 =====")
    print(text[:200])


if __name__ == "__main__":
    main()
