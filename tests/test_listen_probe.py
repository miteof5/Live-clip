"""listen_probe 单元测试：默认模型名 / 配置解析 / 多段失败降级（不调真实 API）。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from liveclip.listen_probe import parse_probe_json, probe_segments, resolve_listen_config


def test_default_model_is_valid_name():
    """默认模型必须是 DashScope 真实存在的名称（无裸 qwen3.5-omni）。"""
    lc = resolve_listen_config({})
    assert lc["model"] == "qwen3.5-omni-flash"
    assert lc["model"] != "qwen3.5-omni"          # 裸名是坑（SKILL 踩坑记录）


def test_config_override_default():
    """config.json listen.model 覆盖默认。"""
    lc = resolve_listen_config({"listen": {"model": "qwen3.5-omni-plus", "api_key": "k"}})
    assert lc["model"] == "qwen3.5-omni-plus"
    assert lc["api_key"] == "k"


def test_parse_probe_json_tolerates_fence():
    """容忍 ```json 代码块包裹。"""
    text = '```json\n{"laughs": true, "description": "观众笑声"}\n```'
    out = parse_probe_json(text)
    assert out["laughs"] is True


def test_parse_probe_json_raises_on_no_json():
    """无 JSON 时显式报错。"""
    try:
        parse_probe_json("我没听懂这段声音")
        assert False, "应抛 ValueError"
    except ValueError:
        pass


def test_probe_segments_degrades_on_error(monkeypatch=None):
    """单段失败降级为 error 条目，后续段继续；成功段保留。"""
    from liveclip import listen_probe

    def boom(*args, **kwargs):
        raise RuntimeError("simulated API failure")

    orig = listen_probe.probe_segment
    if monkeypatch is not None:
        monkeypatch.setattr(listen_probe, "probe_segment", boom)
    else:
        listen_probe.probe_segment = boom
    try:
        cfg = {"listen": {"api_key": "k", "model": "m", "base_url": "http://x"}}
        res = listen_probe.probe_segments("a.wav", [(1.0, 3.0), (5.0, 7.0)], cfg=cfg)
        assert len(res["probes"]) == 2
        assert res["failed"] == 2
        assert all("error" in p for p in res["probes"])
    finally:
        if monkeypatch is None:
            listen_probe.probe_segment = orig


def test_probe_segments_keeps_successes(monkeypatch=None):
    """一半成功一半失败：成功条目保留原字段。"""
    from liveclip import listen_probe

    calls = {"n": 0}

    def fake_probe(audio, start, end, **kw):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("boom")
        return {"t": start, "end": end, "laughs": True, "model": kw["model"]}

    orig = listen_probe.probe_segment
    if monkeypatch is not None:
        monkeypatch.setattr(listen_probe, "probe_segment", fake_probe)
    else:
        listen_probe.probe_segment = fake_probe
    try:
        cfg = {"listen": {"api_key": "k", "model": "m", "base_url": "http://x"}}
        res = listen_probe.probe_segments("a.wav", [(1.0, 3.0), (5.0, 7.0)], cfg=cfg)
        assert res["failed"] == 1
        assert res["probes"][0]["laughs"] is True      # 成功段保留
        assert "error" in res["probes"][1]             # 失败段降级
    finally:
        if monkeypatch is None:
            listen_probe.probe_segment = orig


if __name__ == "__main__":
    test_default_model_is_valid_name()
    test_config_override_default()
    test_parse_probe_json_tolerates_fence()
    test_parse_probe_json_raises_on_no_json()
    test_probe_segments_degrades_on_error(None)
    test_probe_segments_keeps_successes(None)
    print("\nALL LISTEN_PROBE TESTS PASSED")
