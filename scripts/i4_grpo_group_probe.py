import runpy
from pathlib import Path

import torch

root = Path(__file__).resolve().parents[1]
cot_namespace = runpy.run_path(
    str(root / "scripts/i4_cot_behavior_eval.py")
)
baseline_namespace = runpy.run_path(
    str(root / "scripts/i4_behavior_baseline.py")
)

split_cot_response = cot_namespace["split_cot_response"]
has_complete_cot_format = cot_namespace["has_complete_cot_format"]
evaluate_case = baseline_namespace["evaluate_case"]

def reward_response(response, expected_answer):
    _, final_answer = split_cot_response(response)
    answer_correct, numeric_format, _, _ = evaluate_case(
        {"expected_exact": expected_answer},
        final_answer,
    )

    complete = (
        has_complete_cot_format(response)
        and numeric_format
    )
    reward = float(answer_correct and complete)
    return reward, final_answer

expected_answer = "222"
responses = [
    "<think>\n30 × 6 = 180，7 × 6 = 42。\n180 + 42 = 222。\n</think>\n\n222",
    "<think>\n37 × 6 = 226。\n</think>\n\n226",
    "<think>\n37 × 3 = 111。\n111 × 2 = 222。\n</think>\n\n222",
    "<think>\n37 × 6 = 220。\n</think>\n\n220",
]

results = [
    reward_response(response, expected_answer)
    for response in responses
]
rewards = torch.tensor(
    [reward for reward, _ in results],
    dtype=torch.float32,
)

mean = rewards.mean()
std = rewards.std(unbiased=False)
advantages = (rewards - mean) / (std + 1e-8)


for index, ((reward, final_answer), advantage) in enumerate(
    zip(results, advantages)
):
    print(
        f"response {index}: "
        f"answer={final_answer}, "
        f"reward={reward:.1f}, "
        f"advantage={advantage.item():.4f}"
    )

print(f"group mean reward: {mean.item():.4f}")
print(f"group std: {std.item():.4f}")
print(f"has learning signal: {bool(std > 1e-8)}")

assert rewards.tolist() == [1.0, 0.0, 1.0, 0.0]
assert torch.allclose(
    advantages,
    torch.tensor([1.0, -1.0, 1.0, -1.0]),
)
