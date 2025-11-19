import os
import sys
import platform
import importlib
import ensurepip
import subprocess
from pathlib import Path


def ensure_native_deps():
    ensure_one("cybotrade", "2.0.14")
    ensure_one("flow", "1.0.9")
    ensure_one("aion", "0.1.0")


def ensure_one(pkg: str, version: str):
    if _is_installed(pkg):
        return

    wheel_path = resolve_wheel(pkg, version)
    print(f"Installing {pkg}=={version}")
    install_wheel(wheel_path)
    print(f"Installed {pkg}=={version}")


def _is_installed(pkg: str) -> bool:
    try:
        importlib.import_module(pkg)
        return True
    except ImportError:
        return False


def resolve_wheel(pkg: str, version: str) -> Path:
    """
    Attempts to find a matching wheel by iterating over all
    compatible sys_tags. This gives reliable fallback behavior.
    """
    for wheel_filename in derive_possible_wheels(pkg, version):
        wheel_path = download_wheel(pkg, version, wheel_filename)
        if wheel_path is not None:
            return wheel_path

    raise RuntimeError(
        f"No compatible wheel found for {pkg}={version}. "
        "Check that your platform and python version is supported."
    )


def derive_possible_wheels(pkg: str, version: str) -> list[str]:
    """
    Derives a packaging tag for the current system.
    """
    system = platform.system().lower()
    arch = platform.machine().lower()
    pyver = f"cp{sys.version_info.major}{sys.version_info.minor}"

    wheels = []
    match system:
        case "linux":
            wheels = [
                f"{pkg}-{version}-{pyver}-{pyver}-manylinux_2_17_{arch}.manylinux2014_{arch}.whl"
            ]
        case "darwin":
            wheels = [
                f"{pkg}-{version}-{pyver}-{pyver}-macosx_10_12_x86_64.macosx_11_0_arm64.macosx_10_12_universal2.whl"
            ]
        case "windows":
            wheels = [f"{pkg}-{version}-{pyver}-{pyver}-win_{arch}.whl"]

    return wheels


def download_wheel(pkg: str, version: str, filename: str) -> Path | None:
    """ """
    WHEEL_DIR = Path(__file__).parent / "wheels"
    os.makedirs(WHEEL_DIR, exist_ok=True)

    wheel_path = WHEEL_DIR / filename
    if wheel_path.exists():
        return wheel_path

    return None


def install_wheel(wheel_path: Path):
    """
    Install a wheel from a local file, ensuring pip is available.
    """
    try:
        # Try pip first
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-deps",
                str(wheel_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        if "No module named pip" in e.stderr:
            print("[adrs] pip not found, bootstrapping pip...")
            ensurepip.bootstrap()
            subprocess.check_call(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--no-deps",
                    str(wheel_path),
                ]
            )
        else:
            raise RuntimeError(f"Failed to install {wheel_path}: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Failed to install {wheel_path}: {e}") from e
