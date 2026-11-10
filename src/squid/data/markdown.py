import re


CODE_BLOCK_RE = re.compile(r"(```.*?```)", re.DOTALL)
IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")


def clean_text_part(text):
    def replace_image(match):
        alt = match.group(1).strip()
        return f"[图片：{alt}]" if alt else ""

    text = IMAGE_RE.sub(replace_image, text)
    text = LINK_RE.sub(r"\1", text)
    return re.sub(r"\n{3,}", "\n\n", text)


def clean_markdown(text):
    parts = CODE_BLOCK_RE.split(text)

    # 偶数下标是普通文字，奇数下标是完整代码块。
    for index in range(0, len(parts), 2):
        parts[index] = clean_text_part(parts[index])

    return "".join(parts).strip()
