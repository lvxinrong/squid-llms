import runpy
from pathlib import Path

from squid.eval.pairwise import (
    PAIRWISE_PROMPT_V2,
    build_pairwise_prompt,
    build_pairwise_prompt_v2,
)


namespace = runpy.run_path(str(
    Path(__file__).resolve().parents[1]
    / "scripts/i5_pairwise_judge.py"
))

PROMPT_VERSION = namespace["PROMPT_VERSION"]
PROMPT_BUILDERS = namespace["PROMPT_BUILDERS"]


def test_pairwise_cli_registers_both_prompt_versions():
    assert PROMPT_BUILDERS == {
        PROMPT_VERSION: build_pairwise_prompt,
        PAIRWISE_PROMPT_V2: build_pairwise_prompt_v2,
    }
    assert PROMPT_VERSION == "i5-pairwise-prompt-v1"
    assert PAIRWISE_PROMPT_V2 == "i5-pairwise-prompt-v2"
