#!python
from os import getcwd, popen
from os.path import relpath
from packaging.version import Version
import re
import sys
from typing import TextIO


def main():
    repo_dir = popen("git rev-parse --show-toplevel", mode="r").read().rstrip()
    version_path = relpath(f'{repo_dir}/python/lolopy/version.py', getcwd())

    try:
        with open(version_path, "r") as fh:
            new_version = extract_version(fh)
    except Exception as e:
        raise ValueError(f"Couldn't extract version from {version_path}") from e

    try:
        with popen(f"git fetch origin && git show origin/main:python/lolopy/version.py", mode="r") as fh:
            old_version = extract_version(fh)
    except Exception as e:
        raise ValueError(f"Couldn't extract version from main branch") from e

    if new_version.major != old_version.major:
        number = "major"
        code = 1
    elif new_version.minor != old_version.minor:
        number = "minor"
        code = 2
    elif new_version.micro != old_version.micro:
        number = "patch"
        code = 3
    else:
        number = "other component of"
        code = 5

    if new_version > old_version:
        print(f"{number} version bump")
        return 0
    elif new_version < old_version:
        print(f"error - {number} version decreased! {new_version} < {old_version}")
        return code
    else:
        print(f"error - version unchanged! {new_version} == {old_version}")
        return 4


def extract_version(handle: TextIO) -> Version:
    text = handle.read()
    if not re.search(r"\S", text):
        raise ValueError(f"No content")
    match = re.search(r"""^\s*__version__\s*=\s*(['"])(\S+)\1""", text, re.MULTILINE)
    if match:
        return Version(match.group(2))
    else:
        raise ValueError(f"No version found\n{text}")


if __name__ == "__main__":
    sys.tracebacklimit = 0
    result = main()
    if result != 0:
        sys.exit(result)