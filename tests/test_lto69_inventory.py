"""Purpose: Guard LTO-69's complete skrl ownership and compatibility inventory.

Usage: Run ``python3 -m pytest -q tests/test_lto69_inventory.py`` from the
supported CPU environment after adding or reclassifying a Python surface.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parents[1]
BASELINE_FILE_COUNT = 532
AUDIT_TEST = Path("tests/test_lto69_inventory.py")
OWNED_ROOTS = ("skrl/", "tests/", "examples/", "docs/source/")
FORK_DELTA = {
    "skrl/agents/torch/ppo/ppo_rnn.py",
    "skrl/agents/torch/sac/__init__.py",
    "skrl/agents/torch/sac/_common.py",
    "skrl/agents/torch/sac/_factorized.py",
    "skrl/agents/torch/sac/discrete_sac.py",
    "skrl/agents/torch/sac/discrete_sac_cfg.py",
    "skrl/agents/torch/sac/discrete_sac_cfg_factorized.py",
    "skrl/agents/torch/sac/discrete_sac_factorized_simple.py",
    "skrl/memories/torch/__init__.py",
    "skrl/memories/torch/replay.py",
    "skrl/memories/torch/rollout.py",
    "skrl/trainers/torch/base.py",
    "tests/agents/torch/test_discrete_sac.py",
    "tests/agents/torch/test_discrete_sac_factorized_simple.py",
}
PATHSIM_BOUNDARY = {
    "skrl/agents/torch/sac/_common.py",
    "skrl/agents/torch/sac/_factorized.py",
    "skrl/agents/torch/sac/discrete_sac.py",
    "skrl/agents/torch/sac/discrete_sac_cfg.py",
    "skrl/agents/torch/sac/discrete_sac_cfg_factorized.py",
    "skrl/agents/torch/sac/discrete_sac_factorized_simple.py",
    "skrl/memories/torch/replay.py",
    "skrl/memories/torch/rollout.py",
}


def _inventory_paths() -> set[Path]:
    return {
        path.relative_to(ROOT)
        for root in OWNED_ROOTS
        for path in (ROOT / root).rglob("*.py")
        if path.relative_to(ROOT) != AUDIT_TEST
    }


def _is_abstract(path: Path) -> bool:
    return path.name == "__init__.py" or path.name == "base.py"


def _is_gpu_required(path: Path) -> bool:
    return "isaacgym" in path.as_posix().lower() or "isaaclab" in path.as_posix().lower()


def _classify(path: Path) -> dict[str, str | bool]:
    path_string = path.as_posix()
    if path_string.startswith("skrl/"):
        owner = "library"
    elif path_string.startswith("tests/"):
        owner = "test"
    elif path_string.startswith("examples/"):
        owner = "example"
    else:
        owner = "documentation-snippet"
    return {
        "owner": owner,
        "abstract": _is_abstract(path),
        "upstream_status": "fork-modified" if path_string in FORK_DELTA else "upstream-compatible",
        "pathsim_boundary": path_string in PATHSIM_BOUNDARY,
        "device": "gpu-required" if _is_gpu_required(path) else "cpu-capable",
    }


def test_snapshot_has_an_owner_and_device_classification() -> None:
    paths = _inventory_paths()
    classifications = {_path: _classify(_path) for _path in paths}

    assert len(paths) == BASELINE_FILE_COUNT
    assert all(path.as_posix().startswith(OWNED_ROOTS) for path in paths)
    assert {entry["owner"] for entry in classifications.values()} == {
        "library",
        "test",
        "example",
        "documentation-snippet",
    }
    assert {entry["upstream_status"] for entry in classifications.values()} == {
        "fork-modified",
        "upstream-compatible",
    }
    assert {entry["device"] for entry in classifications.values()} == {"cpu-capable", "gpu-required"}


def test_fork_and_pathsim_sets_are_complete_snapshot_members() -> None:
    paths = {path.as_posix() for path in _inventory_paths()}

    assert FORK_DELTA <= paths
    assert PATHSIM_BOUNDARY <= FORK_DELTA
    assert len(FORK_DELTA) == 14
    assert len(PATHSIM_BOUNDARY) == 8
