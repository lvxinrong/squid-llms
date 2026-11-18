import pytest

from squid.eval.bootstrap import paired_bootstrap, build_paired_outcomes


def case(index, passed):
    return {
        "index": index,
        "passed": passed,
    }


def test_outcomes_align_cases_by_index_not_input_order():
    left = [
        case(0, True),
        case(1, True),
        case(2, False),
    ]
    right = [
        case(2, True),
        case(0, False),
        case(1, True),
    ]

    assert build_paired_outcomes(left, right) == [1, 0, -1]


def test_bootstrap_is_reproducible_and_preserves_observed_counts():
    left = [
        case(0, True),
        case(1, False),
        case(2, True),
        case(3, False),
    ]
    right = [
        case(0, False),
        case(1, True),
        case(2, True),
        case(3, False),
    ]

    first = paired_bootstrap(
        left,
        right,
        n_resamples=100,
        seed=7,
    )
    second = paired_bootstrap(
        left,
        right,
        n_resamples=100,
        seed=7,
    )

    assert first == second
    assert first["observed"]["wins"] == 1
    assert first["observed"]["losses"] == 1
    assert first["observed"]["ties"] == 2
    assert first["observed"]["win_rate_all"] == 0.25
    assert first["observed"]["net_win_rate"] == 0
    assert first["observed"]["decisive_win_rate"] == 0.5


def test_all_wins_produce_degenerate_intervals():
    left = [case(index, True) for index in range(3)]
    right = [case(index, False) for index in range(3)]

    result = paired_bootstrap(
        left,
        right,
        n_resamples=100,
        seed=7,
    )

    assert result["intervals"]["win_rate_all"] == {
        "low": 1,
        "high": 1,
    }
    assert result["intervals"]["net_win_rate"] == {
        "low": 1,
        "high": 1,
    }
    assert result["intervals"]["decisive_win_rate"] == {
        "low": 1,
        "high": 1,
    }
    assert result["decisive_valid_resamples"] == 100


def test_all_ties_leave_decisive_interval_undefined():
    left = [case(index, index % 2 == 0) for index in range(4)]
    right = [case(index, index % 2 == 0) for index in range(4)]

    result = paired_bootstrap(
        left,
        right,
        n_resamples=100,
        seed=7,
    )

    assert result["observed"]["decisive_win_rate"] is None
    assert result["intervals"]["decisive_win_rate"] is None
    assert result["decisive_valid_resamples"] == 0
    assert result["decisive_undefined_resamples"] == 100
    assert result["intervals"]["win_rate_all"] == {
        "low": 0,
        "high": 0,
    }


def test_mismatched_case_indices_are_rejected():
    with pytest.raises(ValueError, match="index 必须一致"):
        paired_bootstrap(
            [case(0, True)],
            [case(1, False)],
            n_resamples=10,
        )
