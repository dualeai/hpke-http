#!/usr/bin/env python3
"""Inject one exact tag-derived version into every coordinated release surface."""

from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess
import sys
from collections.abc import Iterable

ROOT = pathlib.Path(__file__).resolve().parent.parent
STABLE_TAG = re.compile(r"^v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")


def run(*arguments: str) -> str:
    return subprocess.run(
        arguments,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def replace_section_version(path: pathlib.Path, section: str, version: str) -> None:
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    section_header = f"[{section}]"
    in_section = False
    replacements = 0
    pattern = re.compile(r'^(\s*version\s*=\s*)"[^"]+"(\s*(?:#.*)?(?:\r?\n)?)$')

    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_section = stripped == section_header
            continue
        if in_section and (match := pattern.fullmatch(line)) is not None:
            lines[index] = f'{match.group(1)}"{version}"{match.group(2)}'
            replacements += 1

    if replacements != 1:
        raise RuntimeError(
            f"{path}: expected one version in {section_header}, found {replacements}"
        )
    path.write_text("".join(lines), encoding="utf-8")


def replace_package_block_versions(
    path: pathlib.Path,
    package_names: Iterable[str],
    version: str,
    *,
    required_source: str | None = None,
) -> None:
    text = path.read_text(encoding="utf-8")
    names = set(package_names)
    replacements = {name: 0 for name in names}
    blocks = re.split(r"(?=^\[\[package\]\]\s*$)", text, flags=re.MULTILINE)

    for index, block in enumerate(blocks):
        name_match = re.search(r'^name\s*=\s*"([^"]+)"\s*$', block, re.MULTILINE)
        if name_match is None or name_match.group(1) not in names:
            continue
        if required_source is not None and required_source not in block:
            continue
        updated, count = re.subn(
            r'^(version\s*=\s*)"[^"]+"\s*$',
            rf'\g<1>"{version}"',
            block,
            count=1,
            flags=re.MULTILINE,
        )
        if count != 1:
            raise RuntimeError(
                f"{path}: package {name_match.group(1)} has no literal version"
            )
        blocks[index] = updated
        replacements[name_match.group(1)] += 1

    invalid = {name: count for name, count in replacements.items() if count != 1}
    if invalid:
        raise RuntimeError(
            f"{path}: expected one local package block per name, got {invalid}"
        )
    path.write_text("".join(blocks), encoding="utf-8")


def update_json_versions(version: str) -> None:
    package_path = ROOT / "typescript/package.json"
    package = json.loads(package_path.read_text(encoding="utf-8"))
    package["version"] = version
    package_path.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")

    lock_path = ROOT / "typescript/package-lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["version"] = version
    lock["packages"][""]["version"] = version
    lock_path.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")


def validate_cargo(version: str) -> None:
    metadata = json.loads(
        run("cargo", "metadata", "--locked", "--format-version", "1", "--no-deps")
    )
    workspace_packages = {
        package["name"]: package["version"]
        for package in metadata["packages"]
        if package["id"] in metadata["workspace_members"]
    }
    expected = {"hpke-http", "hpke-http-py", "hpke-http-wasm"}
    if {name for name in expected if workspace_packages.get(name) != version}:
        raise RuntimeError(
            f"Cargo workspace versions are not coherent: {workspace_packages}"
        )


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: prepare-release-version.py vMAJOR.MINOR.PATCH")

    tag = sys.argv[1]
    match = STABLE_TAG.fullmatch(tag)
    if match is None:
        raise SystemExit(f"release tag must be an exact stable SemVer: {tag}")
    version = tag.removeprefix("v")

    head_sha = run("git", "rev-parse", "HEAD")
    tag_sha = run("git", "rev-parse", f"{tag}^{{commit}}")
    if head_sha != tag_sha:
        raise SystemExit(
            f"tag {tag} resolves to {tag_sha}, not checked-out commit {head_sha}"
        )
    if (workflow_sha := os.environ.get("GITHUB_SHA")) and workflow_sha != head_sha:
        raise SystemExit(
            f"workflow commit {workflow_sha} does not match checked-out commit {head_sha}"
        )

    derived = run("bash", "cicd/version.sh", "-g", ".")
    if derived != version:
        raise SystemExit(
            f"cicd/version.sh derived {derived}, but exact tag {tag} requires {version}"
        )

    replace_section_version(ROOT / "Cargo.toml", "workspace.package", version)
    replace_section_version(ROOT / "python/pyproject.toml", "project", version)
    replace_package_block_versions(
        ROOT / "Cargo.lock",
        ("hpke-http", "hpke-http-py", "hpke-http-wasm"),
        version,
    )
    replace_package_block_versions(
        ROOT / "python/uv.lock",
        ("hpke-http",),
        version,
        required_source='source = { editable = "." }',
    )
    update_json_versions(version)
    validate_cargo(version)
    package = json.loads((ROOT / "typescript/package.json").read_text(encoding="utf-8"))
    package_lock = json.loads(
        (ROOT / "typescript/package-lock.json").read_text(encoding="utf-8")
    )
    if (
        package["version"] != version
        or package_lock["version"] != version
        or package_lock["packages"][""]["version"] != version
    ):
        raise RuntimeError("npm release versions are not coherent")

    changed = set(run("git", "diff", "--name-only").splitlines())
    expected_changes = {
        "Cargo.lock",
        "Cargo.toml",
        "python/pyproject.toml",
        "python/uv.lock",
        "typescript/package-lock.json",
        "typescript/package.json",
    }
    unexpected = changed - expected_changes
    if unexpected:
        raise RuntimeError(
            f"release version preparation changed unexpected paths: {sorted(unexpected)}"
        )

    print(version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
