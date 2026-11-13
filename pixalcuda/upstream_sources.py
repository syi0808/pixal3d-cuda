"""Pinned, unmodified upstream sources and explicit native-extension builds."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


NATIVE_PACKAGES = ("nvdiffrast", "cumesh", "flexgemm", "o-voxel", "natten")


def lock():
    return json.loads(Path(__file__).with_name("upstream-lock.json").read_text())


def source_root(value=None):
    return Path(value or os.environ.get("PIXAL_UPSTREAM_ROOT", "~/.cache/pixal3d-cuda/upstream")).expanduser().resolve()


def git(directory, *arguments):
    return subprocess.check_output(["git", "-C", str(directory), *arguments], text=True).strip()


def verify_source(name, root=None):
    directory = source_root(root) / name
    revision = lock()["sources"][name]["revision"]
    try:
        actual = git(directory, "rev-parse", "HEAD")
        changes = git(directory, "status", "--porcelain", "--untracked-files=no")
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"Missing upstream source {name}; run pixal3d-cuda setup-upstream --root {source_root(root)}") from error
    if actual != revision or changes:
        raise ValueError(f"Upstream {name} must be clean at {revision}; refusing modified or mismatched source")
    return directory


def fetch(root=None):
    root = source_root(root)
    root.mkdir(parents=True, exist_ok=True)
    for name, entry in lock()["sources"].items():
        directory = root / name
        if not directory.exists():
            subprocess.run(["git", "init", str(directory)], check=True)
            subprocess.run(["git", "-C", str(directory), "remote", "add", "origin", entry["url"]], check=True)
            subprocess.run(["git", "-C", str(directory), "fetch", "--depth", "1", "origin", entry["revision"]], check=True)
            subprocess.run(["git", "-C", str(directory), "checkout", "--detach", "FETCH_HEAD"], check=True)
        verify_source(name, root)
        subprocess.run(["git", "-C", str(directory), "submodule", "update", "--init", "--recursive"], check=True)
    return root


def build(root=None, cuda_arch="8.9", package=None):
    if cuda_arch not in ("8.0", "8.6", "8.9"):
        raise ValueError("Native baseline builds target SM80/86/89; T4 SM75 is not a supported BF16 baseline")
    if package is not None and package not in NATIVE_PACKAGES:
        raise ValueError("Unknown native package")
    root = source_root(root)
    for name in lock()["sources"]:
        verify_source(name, root)
    environment = {**os.environ, "TORCH_CUDA_ARCH_LIST": cuda_arch,
                   "NATTEN_CUDA_ARCH": cuda_arch, "MAX_JOBS": "2", "NATTEN_N_WORKERS": "2"}
    packages = {"nvdiffrast": str(root / "nvdiffrast"), "cumesh": str(root / "cumesh"),
                "flexgemm": str(root / "flexgemm"), "o-voxel": str(root / "trellis2" / "o-voxel"),
                "natten": "natten==0.21.0"}
    for name in (NATIVE_PACKAGES if package is None else (package,)):
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-build-isolation", packages[name]], env=environment, check=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root")
    parser.add_argument("--build", action="store_true", help="Compile native CUDA extensions; never starts a GPU job")
    parser.add_argument("--build-package", choices=NATIVE_PACKAGES, help="Build one extension in an independently cached image layer")
    parser.add_argument("--cuda-arch", default="8.9")
    arguments = parser.parse_args(argv)
    root = fetch(arguments.root)
    if arguments.build or arguments.build_package:
        build(root, arguments.cuda_arch, arguments.build_package)
    print(root)


if __name__ == "__main__":
    main()
