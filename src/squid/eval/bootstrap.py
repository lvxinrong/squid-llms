import math
import random


METRICS = (
    "win_rate_all",
    "loss_rate_all",
    "tie_rate",
    "net_win_rate",
    "decisive_win_rate",
)


def build_paired_outcomes(left_cases, right_cases):
    if not left_cases or len(left_cases) != len(right_cases):
        raise ValueError("左右 case 数量必须相同且非空")

    left_by_index = {
        case["index"]: case
        for case in left_cases
    }
    right_by_index = {
        case["index"]: case
        for case in right_cases
    }

    if len(left_by_index) != len(left_cases):
        raise ValueError("左侧 case index 必须唯一")
    if len(right_by_index) != len(right_cases):
        raise ValueError("右侧 case index 必须唯一")
    if set(left_by_index) != set(right_by_index):
        raise ValueError("左右 case index 必须一致")

    outcomes = []

    for index in sorted(left_by_index):
        left = left_by_index[index]
        right = right_by_index[index]

        if left["passed"] and not right["passed"]:
            outcomes.append(1)
        elif right["passed"] and not left["passed"]:
            outcomes.append(-1)
        else:
            outcomes.append(0)

    return outcomes


def summarize_outcomes(outcomes):
    total = len(outcomes)
    wins = sum(outcome == 1 for outcome in outcomes)
    losses = sum(outcome == -1 for outcome in outcomes)
    ties = sum(outcome == 0 for outcome in outcomes)
    decisive = wins + losses

    return {
        "total": total,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_rate_all": wins / total,
        "loss_rate_all": losses / total,
        "tie_rate": ties / total,
        "net_win_rate": (wins - losses) / total,
        "decisive_win_rate": (
            wins / decisive
            if decisive
            else None
        ),
    }


def percentile(values, probability):
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return ordered[lower]

    fraction = position - lower
    return (
        ordered[lower] * (1 - fraction)
        + ordered[upper] * fraction
    )


def paired_bootstrap(
    left_cases,
    right_cases,
    n_resamples=2000,
    seed=20261119,
    confidence=0.95,
):
    if n_resamples < 1:
        raise ValueError("n_resamples 必须大于 0")
    if not 0 < confidence < 1:
        raise ValueError("confidence 必须位于 0 和 1 之间")

    outcomes = build_paired_outcomes(left_cases, right_cases)
    observed = summarize_outcomes(outcomes)
    rng = random.Random(seed)
    samples = {metric: [] for metric in METRICS}
    decisive_valid = 0

    for _ in range(n_resamples):
        resampled = [
            outcomes[rng.randrange(len(outcomes))]
            for _ in range(len(outcomes))
        ]
        summary = summarize_outcomes(resampled)

        for metric in METRICS:
            value = summary[metric]
            if value is not None:
                samples[metric].append(value)

        if summary["decisive_win_rate"] is not None:
            decisive_valid += 1

    lower_probability = (1 - confidence) / 2
    upper_probability = 1 - lower_probability
    intervals = {}

    for metric, values in samples.items():
        intervals[metric] = (
            {
                "low": percentile(values, lower_probability),
                "high": percentile(values, upper_probability),
            }
            if values
            else None
        )

    return {
        "method": "paired case bootstrap percentile",
        "n_resamples": n_resamples,
        "seed": seed,
        "confidence": confidence,
        "observed": observed,
        "intervals": intervals,
        "decisive_valid_resamples": decisive_valid,
        "decisive_undefined_resamples": (
            n_resamples - decisive_valid
        ),
        "boundary": (
            "区间反映当前 case 集合的配对重采样波动，"
            "不消除小样本、数据偏差或基准覆盖不足。"
        ),
    }
