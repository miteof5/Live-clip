"""config 模块测试：配置加载 / ASR 后端规范化 / API key 解析 / ffmpeg 探测。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from liveclip import config as cfg


def test_default_config_merge(tmp_path):
    """未配置项取默认值，配置项覆盖默认。"""
    p = tmp_path / "config.json"
    p.write_text(json.dumps({"silence_noise_db": -40.0}), encoding="utf-8")
    conf = cfg.load_config(p)
    assert conf["silence_noise_db"] == -40.0
    assert conf["silence_min_duration"] == 0.8          # 默认
    assert conf["_config_path"] == str(p)


def test_load_config_missing(tmp_path):
    """缺失 config.json 应抛 ConfigError。"""
    missing = tmp_path / "nope.json"
    try:
        cfg.load_config(missing)
        assert False, "应抛 ConfigError"
    except cfg.ConfigError:
        pass


def test_normalize_backend_defaults():
    """未指定 api_mode 时：bailian → native，其余 → openai。"""
    cfg_data = {}
    b = cfg.normalize_backend({"name": "x", "provider": "bailian", "model": "m"}, cfg_data)
    assert b["api_mode"] == "native"
    b2 = cfg.normalize_backend({"name": "y", "provider": "siliconflow", "model": "m"}, cfg_data)
    assert b2["api_mode"] == "openai"


def test_resolve_api_key_explicit_env(monkeypatch):
    """api_key_env 指定后只从该环境变量读取。"""
    monkeypatch.setenv("MY_KEY", "secret")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "wrong-key")
    backend = {"api_key": "", "api_key_env": "MY_KEY"}
    assert cfg.resolve_api_key(backend, {}) == "secret"


def test_load_asr_backends_active_order(monkeypatch):
    """active 列表决定使用顺序（差→好）；未列出的排后面。"""
    monkeypatch.setenv("DASHSCOPE_API_KEY", "k")
    conf = {
        "asr": {
            "active": ["good", "bad"],
            "backends": [
                {"name": "bad", "provider": "bailian", "model": "m-bad"},
                {"name": "good", "provider": "bailian", "model": "m-good"},
            ],
        }
    }
    bs = cfg.load_asr_backends(conf)
    assert [b["name"] for b in bs] == ["good", "bad"]


def test_find_ffmpeg_from_config():
    """显式配置的 ffmpeg_path 优先（config.json 已指向 tools/ffmpeg）。"""
    ff = cfg.find_ffmpeg()
    assert ff, "本机应能找到 ffmpeg（config.json 已配置 tools/ffmpeg/ffmpeg.exe）"
    assert Path(ff).exists()


if __name__ == "__main__":
    import tempfile
    import os
    from pathlib import Path as P

    class T:  # 简化 tmp_path（手动运行时）
        pass
    import shutil
    d = tempfile.mkdtemp()
    try:
        test_default_config_merge(P(d))
        test_load_config_missing(P(d))
        test_normalize_backend_defaults()
        os.environ["MY_KEY"] = "secret"
        test_resolve_api_key_explicit_env(type("M", (), {}))
        test_load_asr_backends_active_order(type("M", (), {}))
        test_find_ffmpeg_from_config()
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print("\nALL CONFIG TESTS PASSED")
