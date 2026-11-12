"""I1 旧数据管道基线：一次性读取、编码全部语料并构建 Python token 窗口列表。

用途：记录 legacy TokenStreamDataset 的耗时与内存，为后续 memmap 对照提供参照。
输入：data/blog/train.txt 和 Qwen tokenizer；报告写入 data/reports/i1-loader-baseline.json。
边界：tracemalloc 与 ru_maxrss 口径不同；本脚本测量数据准备，不测量模型训练吞吐或效果。
"""

import json
import resource
import time
import tracemalloc
from pathlib import Path

from squid.data.token_stream import TokenStreamDataset, pack_tokens
from squid.tokenizer.qwen_tokenizer import QwenTokenizerAdapter

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data/blog/train.txt"
OUTPUT_PATH = ROOT / "data/reports/i1-loader-baseline.json"

def main():
    tokenizer = QwenTokenizerAdapter()
    text = TRAIN_PATH.read_text(encoding="utf-8")

    tracemalloc.start()

    start = time.perf_counter()
    ids = pack_tokens(
        [text],
        tokenizer,
        tokenizer.eos_id,
    )
    pack_seconds = time.perf_counter() - start

    pack_current, _ = tracemalloc.get_traced_memory()

    start = time.perf_counter()
    dataset = TokenStreamDataset(
        ids,
        max_length=128,
        stride=128,
    )
    dataset_seconds = time.perf_counter() - start

    dataset_current, _ = tracemalloc.get_traced_memory()

    # macOS 的 ru_maxrss 单位是 bytes，表示进程历史峰值 RSS，
    # 不是当前 Python 对象的精确大小。
    peak_rss_bytes = resource.getrusage(
        resource.RUSAGE_SELF
    ).ru_maxrss

    report = {
        "tokens_including_eos": len(ids),
        "context_length": 128,
        "stride": 128,
        "windows": len(dataset),
        "pack_seconds": pack_seconds,
        "dataset_seconds": dataset_seconds,
        "tracemalloc_after_pack_bytes": pack_current,
        "tracemalloc_after_dataset_bytes": dataset_current,
        "process_peak_rss_bytes": peak_rss_bytes,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2))
    print(f"\n基线已保存到：{OUTPUT_PATH}")

    # 保留引用，避免在统计前被垃圾回收。
    assert dataset.input_ids


if __name__ == "__main__":
    main()