# Squid

一只从小养到大的 LLM。

从《从零构建大模型》的手搓代码出发，按工程结构重写的大模型实现：分词、模型、训练、生成、评估，全部自包含，每一行都自己搞懂。

## 安装

```bash
uv pip install -e .
```

## 结构

```
src/squid/
├── config.py      # 超参唯一入口
├── device.py      # device 唯一入口（CUDA > MPS > CPU）
├── tokenizer/     # 编解码
├── model/         # GPT 模型与组件
├── data/          # 数据管道
├── train/         # 预训练 / 分类微调 / 指令微调
├── generate/      # 文本生成
├── eval/          # 评估（含 LLM 评委）
└── checkpoints/   # 权重存档的加载与保存
```

学习记录见 [blog.lvxinrong.com/llm/](https://blog.lvxinrong.com/llm/)。
