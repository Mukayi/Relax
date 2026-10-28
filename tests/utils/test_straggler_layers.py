# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""The profiler's layers import only the layers before them: ``core`` the
standard library, numpy and itself; ``megatron_adapter`` also Megatron;
``relax_integration`` anything."""

import ast
import subprocess
import sys
from pathlib import Path

import pytest


_PACKAGE = Path(__file__).resolve().parents[2] / "relax" / "utils" / "straggler"
_CORE = "relax.utils.straggler.core"
# requests: the level poller's HTTP client, imported when a poller is created.
_CORE_THIRD_PARTY = frozenset({"numpy", "requests"})


def _imports(path):
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, f"{path.name}: relative imports hide which layer they reach"
            yield node.module


def _is_core(name):
    return name == _CORE or name.startswith(_CORE + ".")


def _disallowed(path, *, third_party=frozenset()):
    return [
        name
        for name in _imports(path)
        if not (name.split(".")[0] in sys.stdlib_module_names or name.split(".")[0] in third_party or _is_core(name))
    ]


@pytest.mark.parametrize("path", sorted((_PACKAGE / "core").glob("*.py")), ids=lambda path: path.name)
def test_the_core_imports_only_the_standard_library_numpy_and_itself(path):
    assert _disallowed(path, third_party=_CORE_THIRD_PARTY) == []


def test_the_megatron_adapter_imports_only_megatron_and_the_core():
    assert _disallowed(_PACKAGE / "megatron_adapter.py", third_party=frozenset({"megatron"})) == []


def test_importing_the_core_the_adapter_or_the_control_routes_loads_neither_torch_nor_megatron():
    modules = [
        *(f"{_CORE}.{path.stem}" for path in sorted((_PACKAGE / "core").glob("*.py")) if path.stem != "__init__"),
        "relax.utils.straggler.megatron_adapter",
        "relax.utils.straggler.control_routes",
    ]
    code = (
        "import importlib, sys\n"
        f"for name in {modules!r}:\n"
        "    importlib.import_module(name)\n"
        "print(sorted({'torch', 'megatron'} & {name.split('.')[0] for name in sys.modules}))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=_PACKAGE.parents[2], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "[]"
