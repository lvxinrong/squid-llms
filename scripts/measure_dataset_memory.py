"""I1 Dataset 内存对照：比较一次性 Python token 列表与 NumPy memmap 的构建成本。

用途：用 tracemalloc、构建耗时和进程峰值 RSS 解释为什么大数据集不应全部物化到内存。
用法：--mode legacy 或 --mode memmap；结果分别写入 data/reports/i1-dataset-memory-<mode>.json。
边界：ru_maxrss 是进程历史峰值而非单个对象精确大小；两种模式应分开运行，避免同进程遗留分配污染对照。
"""

import argparse
import json
import resource
import time
import tracemalloc
from pathlib import Path
import numpy as np
from squid.data.memmap_dataset import MemmapTokenDataset
from squid.data.token_stream import TokenStreamDataset


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data/blog/train.txt"
TOKEN_PATH = ROOT / "data/token_store/blog_v1/train.uint32.bin"
REPORT_DIR = ROOT / "data/reports"


def prepare_legacy_ids():
    return np.memmap(
        TOKEN_PATH,
        dtype="uint32",
        mode="r",
    ).tolist()


def measure(mode):
    if mode == "legacy":
        ids = prepare_legacy_ids()
    else:
        ids = None

    # 准备工作在测量前完成，只统计 Dataset 构建产生的分配。
    tracemalloc.start()

    start = time.perf_counter()

    if mode == "legacy":
        dataset = TokenStreamDataset(
            ids,
            max_length=128,
            stride=128,
        )
    else:
        dataset = MemmapTokenDataset(
            TOKEN_PATH,
            max_length=128,
            stride=128,
        )

    elapsed = time.perf_counter() - start

    current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    peak_rss_bytes = resource.getrusage(
        resource.RUSAGE_SELF
    ).ru_maxrss

    report = {
        "mode": mode,
        "source_tokens": (
            len(ids) if mode == "legacy" else len(dataset.tokens)
        ),
        "windows": len(dataset),
        "dataset_seconds": elapsed,
        "tracemalloc_current_bytes": current_bytes,
        "tracemalloc_peak_bytes": peak_bytes,
        "process_peak_rss_bytes": peak_rss_bytes,
    }

    tracemalloc.stop()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / f"i1-dataset-memory-{mode}.json"
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2))
    print(f"\n报告已保存到：{report_path}")

    # 防止 dataset 在报告生成前被当成无用对象回收。
    assert len(dataset) > 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["legacy", "memmap"],
        required=True,
    )
    args = parser.parse_args()
    measure(args.mode)


if __name__ == "__main__":
    main()
