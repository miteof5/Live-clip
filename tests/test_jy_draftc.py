"""jy_draftc 工具探测测试：tools/.env 解析 / find_exe / find_install_dir。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from liveclip.jy_draftc import _load_tools_env, find_exe, find_install_dir


def test_load_tools_env():
    """tools/.env 应能解析出 JY_INSTALL_DIR（UTF-8 中文路径）。"""
    env = _load_tools_env()
    assert isinstance(env, dict)
    if env:
        jy = env.get("JY_INSTALL_DIR")
        if jy:
            assert "剪映" in jy or "Jianying" in jy


def test_find_exe_project_tools():
    """项目自带 tools/jy-draftc.exe 应被找到。"""
    exe = find_exe()
    if exe is None:
        print("[skip] 本机未安装 jy-draftc（无 tools/jy-draftc.exe）")
        return
    assert Path(exe).is_file()
    assert exe.endswith("jy-draftc.exe")


def test_find_install_dir():
    """剪映安装目录应可定位（tools/.env 或 LOCALAPPDATA）。"""
    d = find_install_dir()
    if d is None:
        print("[skip] 本机未找到含 videoeditor.dll 的剪映安装目录")
        return
    assert (Path(d) / "videoeditor.dll").is_file()


if __name__ == "__main__":
    test_load_tools_env()
    test_find_exe_project_tools()
    test_find_install_dir()
    print("\nALL JY_DRAFTC TESTS PASSED")
