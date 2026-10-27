from squid.tokenizer import count_pairs, merge, train_bpe, decode, encode


def test_count_pairs():
    assert count_pairs([1, 2, 3, 1, 2]) == {(1, 2): 2, (2, 3): 1, (3, 1): 1}

def test_merge():
    assert merge([1, 2, 1, 2, 3], (1, 2), 256) == [256, 256, 3]  # 连续命中
    assert merge([3, 1, 2], (1, 2), 256) == [3, 256]  # 奇数位起配对
    assert merge([1, 1, 1], (1, 1), 256) == [256, 1]  # 重叠对只合一次


def test_train_and_roundtrip():
    text = "aaabdaaabac"
    vocab, merges = train_bpe(text, 3)
    assert merges == {(97, 97): 256, (256, 97): 257, (257, 98): 258}
    assert decode(encode(text, merges), vocab) == text