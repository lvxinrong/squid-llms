"""
1. 收语料：把 lv_blog/content/**/*.md 读出来拼成一份文本（留 2 篇做 held-out，不参与训练）
2. 训练：train_bpe(train_text, 512)，保存 vocab/merges 中间产物
3. 对比：同一段 held-out 文本，你的 BPE 和 TiktokenTokenizer("gpt2") 各 encode 一次，对比 token 数、bytes/token
"""
import random
import tomllib
from pathlib import Path

from squid.tokenizer import train_bpe, save_bpe, load_bpe, decode, encode, TiktokenTokenizer

CONTENT = Path("/Users/lv/lv_repo/lv_blog/content")
OUT = Path(__file__).resolve().parents[1] / "data" / "blog"


def read_post(path):
    """返回 (front matter, 正文)。Hugo 的 front matter 用 +++ 包 TOML。"""
    raw = path.read_text(encoding="utf-8")
    if raw.startswith("+++"):
        end = raw.find("\n+++", 3)
        if end != -1:
            return tomllib.loads(raw[3:end]), raw[end + 4:]
    return {}, raw


def prepare_corpus():
    posts = []
    for path in sorted(CONTENT.rglob("*.md")):
        if path.name.startswith("_"):                # _index.md 是栏目索引，不是文章
            continue
        meta, body = read_post(path)
        body = body.strip()
        if not body or meta.get("draft") is True:    # 草稿不参与训练
            continue
        if meta.get("layout") in {"archives", "columns", "search"}:
            continue
        # 正文里有 <|endoftext|> 字面量，tiktoken 默认 encode 会报错，先替换掉
        body = body.replace("<|endoftext|>", "<|end-of-text|>")
        posts.append((path, meta, body))

    pool = [p for p in posts if "date" in p[1]]
    held = random.Random(42).sample(pool, 2)
    held_paths = {p[0] for p in held}
    train = [p for p in posts if p[0] not in held_paths]

    train_text = "\n\n".join(body for _, _, body in train)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "train.txt").write_text(train_text, encoding="utf-8")
    for path, _, body in held:
        (OUT / f"heldout-{path.stem}.txt").write_text(body, encoding="utf-8")
    print(f"语料：{len(train)} 篇训练 / {len(held)} 篇 held-out，train.txt {len(train_text)} 字符")


def bpe_train(file_path, num_merges=512):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    assert text, "训练文本为空，先跑 prepare_corpus()"
    vocab, merges = train_bpe(text, num_merges)
    return vocab, merges


if __name__ == '__main__':
    # prepare_corpus()
    nm = 512
    vo, mes = bpe_train(OUT / "train.txt", nm)
    save_bpe(vo, mes, OUT / ("m" + str(nm)))
    print(f"保存完成：vocab {len(vo)} 项，merges {len(mes)} 条")
    # vo, mse = load_bpe(OUT)
    # items = list(mse.items())  # dict 3.7+ 保持插入顺序，即 rank 顺序
    # text = open("/Users/lv/lv_repo/squid-llms/data/blog/heldout-手搓LLM笔记07-指令微调.txt", encoding="utf-8").read()
    # tok = TiktokenTokenizer("gpt2")
    # gpt = len(tok.encode(text))
    # mine = len(encode(text, mse))
    #
    # print(f"我的 BPE: {mine} tokens, {len(text.encode('utf-8')) / mine:.2f} bytes/token")
    # print(f"GPT-2:    {gpt} tokens, {len(text.encode('utf-8')) / gpt:.2f} bytes/token")
    #
    # text2 = open("/Users/lv/lv_repo/squid-llms/data/blog/heldout-我想把家里的设备组成一个私人AI实验室.txt", encoding="utf-8").read()
    # tok2 = TiktokenTokenizer("gpt2")
    # gpt2 = len(tok2.encode(text2))
    # mine2 = len(encode(text2, mse))
    #
    # print(f"我的BPE: {mine2} tokens, {len(text2.encode('utf-8')) / mine2:.2f} bytes/token")
    # print(f"GPT-2:    {gpt2} tokens, {len(text2.encode('utf-8')) / gpt2:.2f} bytes/token")
    #
    # print(len(text.encode("utf-8")), mine, gpt)
    # print(len(text2.encode("utf-8")), mine2, gpt2)




