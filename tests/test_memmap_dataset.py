from pathlib import Path

import numpy as np
import torch

from squid.data.memmap_dataset import MemmapTokenDataset
from squid.data.token_stream import TokenStreamDataset


ROOT = Path(__file__).resolve().parents[1]
TOKEN_PATH = (
    ROOT / "data/token_store/blog_v1/train.uint32.bin"
)

def test_memmap_dataset_matches_legacy_dataset():
    for stride in (64, 128):
        memmap_dataset = MemmapTokenDataset(
            TOKEN_PATH,
            max_length=128,
            stride=stride,
        )

        # 仅为测试构造旧实现；生产读取不需要这个 tolist()。
        legacy_dataset = TokenStreamDataset(
            np.memmap(
                TOKEN_PATH,
                dtype="uint32",
                mode="r",
            ).tolist(),
            max_length=128,
            stride=stride,
        )

        assert len(memmap_dataset) == len(legacy_dataset)

        for idx in (0, 1, 17, len(legacy_dataset) - 1):
            old_input, old_target = legacy_dataset[idx]
            new_input, new_target = memmap_dataset[idx]

            assert new_input.dtype == torch.long
            assert new_target.dtype == torch.long

            torch.testing.assert_close(
                new_input,
                old_input,
                rtol=0.0,
                atol=0.0,
            )
            torch.testing.assert_close(
                new_target,
                old_target,
                rtol=0.0,
                atol=0.0,
            )