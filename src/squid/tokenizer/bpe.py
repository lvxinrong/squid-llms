
def count_pairs(ids: list[int]) -> dict[tuple[int, int], int]:
    """统计相邻对频次。例：[1,2,3,1,2] → {(1,2):2, (2,3):1, (3,1):1}"""
    counts = {}
    for a, b in zip(ids, ids[1:]):   # 相邻两两配对，这行是白给的
        key = (a, b)
        if key not in counts:
            counts[key] = 1
        else:
            counts[key] += 1
    return counts# 你写：往 counts 里累加 (a,b)


def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """把 ids 里所有出现的 pair 替换成 new_id，返回新列表。"""
    new_idx = []
    i= 0
    while i < len(ids):
        if i < len(ids) -1 and (ids[i], ids[i + 1]) == pair:
            new_idx.append(new_id)
            i = i + 2
        else:
            new_idx.append(ids[i])
            i = i + 1
    return new_idx


def train_bpe(text: str, num_merges: int):
    ids = list(text.encode("utf-8"))
    vocab = {i: bytes([i]) for i in range(256)}
    merges = {}

    for _ in range(num_merges):
        counts = count_pairs(ids)
        if not counts:
            break
        best = max(counts, key=counts.get)
        new_id = 256 + len(merges)
        ids = merge(ids, best, new_id)
        merges[best] = new_id
        vocab[new_id] = vocab[best[0]] + vocab[best[1]]

    return vocab, merges


def decode(ids: list[int], vocab) -> str:
    return b"".join(vocab[i] for i in ids).decode("utf-8")

def encode(text: str, merges) -> list[int]:
    ids = list(text.encode("utf-8"))
    for pair, new_id in merges.items():
        ids = merge(ids, pair, new_id)
    return ids



