#!/usr/bin/env python3
"""Download (if needed) and run a local NATS server with JetStream enabled.

Usage:
    uv run python scripts/nats_server.py [--port 4222]
"""

import argparse
import os
import platform
import subprocess
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve
from urllib.error import URLError

NATS_VERSION = "2.10.22"
BIN_DIR = Path(__file__).parent.parent / ".nats"


def _os_arch() -> tuple[str, str]:
    sys_name = platform.system().lower()
    machine = platform.machine().lower()
    os_map = {"darwin": "darwin", "linux": "linux", "windows": "windows"}
    arch_map = {
        "x86_64": "amd64",
        "amd64": "amd64",
        "aarch64": "arm64",
        "arm64": "arm64",
    }
    os_name = os_map.get(sys_name)
    arch = arch_map.get(machine)
    if not os_name or not arch:
        print(f"Unsupported platform: {sys_name}/{machine}", file=sys.stderr)
        sys.exit(1)
    return os_name, arch


def ensure_binary() -> Path:
    os_name, arch = _os_arch()
    ext = ".exe" if os_name == "windows" else ""
    binary = BIN_DIR / f"nats-server{ext}"

    if binary.exists():
        return binary

    BIN_DIR.mkdir(parents=True, exist_ok=True)

    filename = f"nats-server-v{NATS_VERSION}-{os_name}-{arch}.zip"
    url = f"https://github.com/nats-io/nats-server/releases/download/v{NATS_VERSION}/{filename}"
    zip_path = BIN_DIR / filename

    print(f"Downloading nats-server v{NATS_VERSION} ({os_name}/{arch})...")
    try:
        urlretrieve(url, zip_path)
    except URLError as e:
        print(f"Download failed: {e}", file=sys.stderr)
        sys.exit(1)

    print("Extracting...")
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            if member.endswith(f"nats-server{ext}") and "/" in member:
                data = zf.read(member)
                binary.write_bytes(data)
                break
    zip_path.unlink(missing_ok=True)

    if os_name != "windows":
        binary.chmod(0o755)

    print(f"Installed: {binary}\n")
    return binary


def main():
    parser = argparse.ArgumentParser(description="Run a local NATS server")
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("NATS_PORT", 4222))
    )
    parser.add_argument("--store-dir", default=str(BIN_DIR / "jetstream"))
    parser.add_argument("--no-jetstream", action="store_true")
    args = parser.parse_args()

    binary = ensure_binary()

    cmd = [str(binary), f"--port={args.port}"]
    if not args.no_jetstream:
        cmd += ["-js", f"-sd={args.store_dir}"]

    print(f"NATS server v{NATS_VERSION}")
    print(f"  port      : {args.port}")
    print(f"  jetstream : {'off' if args.no_jetstream else args.store_dir}")
    print("Press Ctrl+C to stop.\n")

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
