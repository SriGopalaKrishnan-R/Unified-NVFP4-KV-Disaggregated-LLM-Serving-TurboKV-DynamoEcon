"""Single command build and verification runner for C++20 engines, Python control plane, and UI."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BUILD_DIR = ROOT / "build"


def _augment_path() -> dict[str, str]:
    env = os.environ.copy()
    extra_dirs = [
        str(Path(sys.executable).parent / "Scripts"),
        str(Path.home() / "AppData/Roaming/Python" / f"Python{sys.version_info.major}{sys.version_info.minor}" / "Scripts"),
        "C:/msys64/mingw64/bin",
        "C:/msys64/ucrt64/bin",
    ]
    existing = [d for d in extra_dirs if Path(d).exists()]
    if existing:
        env["PATH"] = os.pathsep.join(existing + [env.get("PATH", "")])
    env["PYTHONPATH"] = str(ROOT)
    return env


def run_step(cmd: list[str], cwd: Path, env: dict[str, str]) -> None:
    print(f"\n>>> Running {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def main() -> None:
    env = _augment_path()
    cmake_cmd = ["cmake", "-S", ".", "-B", "build", "-DCMAKE_BUILD_TYPE=Release", "-DUSE_CPU_SIM=ON"]
    if shutil.which("ninja", path=env.get("PATH")):
        cmake_cmd.extend(["-G", "Ninja"])
    mingw_gxx = Path("C:/msys64/mingw64/bin/g++.exe")
    mingw_gcc = Path("C:/msys64/mingw64/bin/gcc.exe")
    if sys.platform == "win32" and mingw_gxx.exists() and not shutil.which("cl"):
        cmake_cmd.extend([
            f"-DCMAKE_CXX_COMPILER={mingw_gxx.as_posix()}",
            f"-DCMAKE_C_COMPILER={mingw_gcc.as_posix()}",
        ])

    run_step(cmake_cmd, ROOT, env)
    run_step(["cmake", "--build", "build", "-j"], ROOT, env)
    run_step(["ctest", "--test-dir", "build", "-V"], ROOT, env)
    run_step([sys.executable, "-m", "pytest", "control_plane/tests/test_control_plane.py", "-v"], ROOT, env)

    npm_bin = shutil.which("npm", path=env.get("PATH"))
    ui_dir = ROOT / "ui"
    if npm_bin and ui_dir.exists():
        if not (ui_dir / "node_modules").exists():
            run_step([npm_bin, "install"], ui_dir, env)
        run_step([npm_bin, "run", "build"], ui_dir, env)

    print("\nAll C++20 tests, Python control plane checks, and UI build completed.")


if __name__ == "__main__":
    main()
