from squid.tokenizer import count_pairs, merge, encode


def sweep(train_text, heldout_text, checkpoints):
    ids = list(train_text.encode("utf-8"))
    vocab = {i: bytes([i]) for i in range(256)}
    merges = {}
    results = []
    for n in range(1, max(checkpoints) + 1):
        counts = count_pairs(ids)
        if not counts:
            break
        best = max(counts, key=counts.get)
        new_id = 256 + len(merges)
        ids = merge(ids, best, new_id)
        merges[best] = new_id
        vocab[new_id] = vocab[best[0]] + vocab[best[1]]
        if n in checkpoints:  # 到检查点，测一次 held-out
            t = len(encode(heldout_text, merges))
            density = len(heldout_text.encode("utf-8")) / t
            results.append((n, t, density))
            print(f"merges={n:5d}  tokens={t:5d}  bytes/token={density:.2f}")
    return results

if __name__ == "__main__":
    checkpoints = [0, 64, 128, 256, 512, 1024, 2000, 4000, 8000, 16000]
    train_text = open("/Users/lv/lv_repo/squid-llms/data/held/train.txt", encoding="utf-8").read()
    held_out = open("/Users/lv/lv_repo/squid-llms/data/held/heldout-手搓LLM笔记07-指令微调.txt", encoding="utf-8").read()
    sweep(train_text, held_out, checkpoints)