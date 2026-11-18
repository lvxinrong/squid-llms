import runpy
from pathlib import Path

from squid.eval.pairwise import build_pairwise_prompt


namespace = runpy.run_path(str(
    Path(__file__).resolve().parents[1]
    / "scripts/i5_length_control.py"
))
build_pairs = namespace["build_pairs"]


def pairs_by_id():
    return {
        pair["pair_id"]: pair
        for pair in build_pairs()
    }


def test_exact_and_padding_controls():
    pairs = pairs_by_id()
    exact = pairs["length-01-exact"]
    padding = pairs["length-02-padding"]

    assert exact["responses"]["A"] == exact["responses"]["B"]
    assert exact["expected_winner"] == "tie"

    assert (
        padding["character_counts"]["B"]
        > padding["character_counts"]["A"]
    )
    assert "还需要再次确认" in padding["responses"]["B"]
    assert "210 + 147" not in padding["responses"]["B"]
    assert padding["expected_winner"] == "A"


def test_informative_and_correctness_controls():
    pairs = pairs_by_id()
    informative = pairs["length-03-informative"]
    correctness = pairs["length-04-correctness"]

    assert (
        informative["character_counts"]["B"]
        > informative["character_counts"]["A"]
    )
    assert "210 + 147" in informative["responses"]["B"]
    assert informative["expected_winner"] is None

    assert correctness["responses"]["A"].endswith("357")
    assert correctness["responses"]["B"].endswith("347")
    assert (
        correctness["character_counts"]["B"]
        > correctness["character_counts"]["A"]
    )
    assert correctness["expected_winner"] == "A"


def test_prompt_does_not_leak_experimental_design():
    pair = pairs_by_id()["length-02-padding"]
    prompt = build_pairwise_prompt(pair)

    assert pair["condition"] not in prompt
    assert pair["roles"]["A"] not in prompt
    assert pair["roles"]["B"] not in prompt
    assert "expected_winner" not in prompt
    assert "[回答 A]" in prompt
    assert "[回答 B]" in prompt
