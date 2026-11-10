from squid.data.markdown import clean_markdown


def test_link_keeps_visible_text():
    text = "请阅读[我的笔记](https://example.com/note)。"
    assert clean_markdown(text) == "请阅读我的笔记。"


def test_image_keeps_alt():
    text = "![实验台](/llm/workbench.png)"
    assert clean_markdown(text) == "[图片：实验台]"


def test_image_without_alt_is_removed():
    assert clean_markdown("![](/llm/workbench.png)") == ""


def test_code_block_is_not_modified():
    text = (
        "先看[说明](https://example.com)。\n\n"
        "```python\n"
        '# [这行注释](不是链接) 和 "![](不是图片)" 都保留\n'
        "```"
    )

    result = clean_markdown(text)

    assert "先看说明。" in result
    assert '# [这行注释](不是链接) 和 "![](不是图片)" 都保留' in result


def test_excess_blank_lines_are_merged():
    assert clean_markdown("第一段\n\n\n\n第二段") == "第一段\n\n第二段"
