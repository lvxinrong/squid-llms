"""极简 .env 加载：把项目根目录 .env 里的键值写进 os.environ。

只做必要的事，不引入额外依赖：
- 支持 `KEY=value`、`export KEY=value`、空行、以 # 开头的注释行
- 支持值两端加单引号或双引号（引号会被去掉）
- 未加引号的值支持行尾注释：`KEY=value  # 说明`
- 默认不覆盖已存在的环境变量（真环境变量优先，.env 只做兜底）

路径不依赖启动时的工作目录，而是按本文件位置找仓库根目录，
所以在 PyCharm 里以任何 Working directory 运行都能读到。
"""

import os
from pathlib import Path

# src/squid/env.py -> src/squid -> src -> 仓库根目录
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_PATH = _PROJECT_ROOT / ".env"


def _parse_line(raw_line):
    """解析一行，返回 (key, value)；空行 / 注释 / 非法行返回 None。"""
    line = raw_line.strip()
    if not line or line.startswith("#"):
        return None

    if line.startswith("export "):
        line = line[len("export "):].lstrip()

    if "=" not in line:
        return None

    key, value = line.split("=", 1)
    key, value = key.strip(), value.strip()
    if not key:
        return None

    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        value = value[1:-1]          # 引号包裹：原样取值
    else:
        value = value.split(" #", 1)[0].rstrip()  # 未加引号：允许行尾注释

    return key, value


def load_env(path=None, override=False):
    """加载 .env 文件到 os.environ。

    Args:
        path: .env 路径，默认仓库根目录下的 .env。
        override: 是否覆盖已存在的环境变量，默认 False。

    Returns:
        bool: 文件存在并被读取返回 True，否则 False。
    """
    env_path = Path(path) if path is not None else _DEFAULT_PATH
    if not env_path.is_file():
        return False

    # utf-8-sig 容忍 Windows / 部分编辑器写入的 BOM
    for raw_line in env_path.read_text(encoding="utf-8-sig").splitlines():
        parsed = _parse_line(raw_line)
        if parsed is None:
            continue
        key, value = parsed
        if override or key not in os.environ:
            os.environ[key] = value
    return True
