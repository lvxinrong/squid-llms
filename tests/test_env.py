import os

import pytest

from squid.env import load_env


@pytest.fixture(autouse=True)
def restore_environ():
    """load_env 直接改 os.environ，测试结束后恢复原样。"""
    before = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(before)


def _write(tmp_path, text):
    env_file = tmp_path / ".env"
    env_file.write_text(text, encoding="utf-8")
    return env_file


def test_load_basic(tmp_path):
    env_file = _write(tmp_path, "SQUID_TEST_KEY=sk-abc\n")
    assert load_env(env_file) is True
    assert os.environ["SQUID_TEST_KEY"] == "sk-abc"


def test_missing_file_returns_false(tmp_path):
    assert load_env(tmp_path / "no-such.env") is False
    assert "SQUID_TEST_KEY" not in os.environ


def test_comments_quotes_export_and_blank_lines(tmp_path):
    env_file = _write(
        tmp_path,
        """
# 注释行
SQUID_TEST_A="双引号值"
SQUID_TEST_B='单引号值'
export SQUID_TEST_C=export 值
SQUID_TEST_D=值  # 行尾注释

不是键值对
""",
    )
    load_env(env_file)
    assert os.environ["SQUID_TEST_A"] == "双引号值"
    assert os.environ["SQUID_TEST_B"] == "单引号值"
    assert os.environ["SQUID_TEST_C"] == "export 值"
    assert os.environ["SQUID_TEST_D"] == "值"


def test_existing_env_wins_by_default(tmp_path, monkeypatch):
    monkeypatch.setenv("SQUID_TEST_KEY", "from-shell")
    load_env(_write(tmp_path, "SQUID_TEST_KEY=from-dotenv\n"))
    assert os.environ["SQUID_TEST_KEY"] == "from-shell"


def test_override_flag(tmp_path, monkeypatch):
    monkeypatch.setenv("SQUID_TEST_KEY", "from-shell")
    load_env(_write(tmp_path, "SQUID_TEST_KEY=from-dotenv\n"), override=True)
    assert os.environ["SQUID_TEST_KEY"] == "from-dotenv"


def test_repo_root_default_path(tmp_path, monkeypatch):
    # 默认路径按文件位置定位仓库根目录，与当前工作目录无关
    from squid import env

    monkeypatch.chdir(tmp_path)
    assert env._DEFAULT_PATH == env._PROJECT_ROOT / ".env"
