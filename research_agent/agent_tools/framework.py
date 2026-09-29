"""Import the sibling Tensor Inpainting Agent Framework source tree in both supported launch modes.

The learning project is intentionally a standalone folder inside the
``tensor_inpainting_agent`` source checkout.  Users may launch it from that checkout as
``python -m research_agent...`` or install/import the parent package.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


def _comes_from(module, package_dir: Path) -> bool:
    module_path = getattr(module, "__file__", None)
    if not module_path:
        return False
    try:
        Path(module_path).resolve().relative_to(package_dir.resolve())
        return True
    except ValueError:
        return False


def _make_namespace_package(name: str, package_dir: Path):
    package = types.ModuleType(name)
    package.__file__ = None
    package.__package__ = name
    package.__path__ = [str(package_dir)]
    sys.modules[name] = package
    return package


def _load_local_package(name: str, package_dir: Path):
    """Load this checkout under its canonical package name, regardless of folder name."""

    init_path = package_dir / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        name,
        init_path,
        submodule_search_locations=[str(package_dir)],
    )
    if spec is None or spec.loader is None:
        raise ImportError("cannot load local package from %s" % init_path)
    package = importlib.util.module_from_spec(spec)
    sys.modules[name] = package
    spec.loader.exec_module(package)
    return package


# The Conda environment may already contain an unrelated PyPI package named
# ``tensor_inpainting_agent``. Locate the sibling framework from this file instead of
# relying on the checkout directory name or a root __init__.py.
local_package_dir = Path(__file__).resolve().parents[2]
repository_parent = local_package_dir.parent
local_framework_files = (
    local_package_dir / "core" / "llm.py",
    local_package_dir / "tools" / "base.py",
    local_package_dir / "observability" / "trace_logger.py",
)
if all(path.is_file() for path in local_framework_files):
    parent_text = str(repository_parent)
    if parent_text in sys.path:
        sys.path.remove(parent_text)
    sys.path.insert(0, parent_text)
    loaded_package = sys.modules.get("tensor_inpainting_agent")
    if loaded_package is not None and not _comes_from(
        loaded_package, local_package_dir
    ):
        for module_name in list(sys.modules):
            if module_name == "tensor_inpainting_agent" or module_name.startswith(
                "tensor_inpainting_agent."
            ):
                sys.modules.pop(module_name, None)

    loaded_package = sys.modules.get("tensor_inpainting_agent")
    if loaded_package is not None and _comes_from(loaded_package, local_package_dir):
        tensor_inpainting_agent = loaded_package
    elif (local_package_dir / "__init__.py").is_file():
        tensor_inpainting_agent = _load_local_package(
            "tensor_inpainting_agent", local_package_dir
        )
    else:
        # A source-only deployment may omit the package marker at the checkout
        # root. Build a minimal namespace package so absolute framework imports
        # still resolve to this checkout rather than an unrelated installed
        # package.
        tensor_inpainting_agent = _make_namespace_package("tensor_inpainting_agent", local_package_dir)
        for subpackage_name in ("core", "tools", "observability"):
            full_name = "tensor_inpainting_agent.%s" % subpackage_name
            subpackage = _make_namespace_package(
                full_name, local_package_dir / subpackage_name
            )
            setattr(tensor_inpainting_agent, subpackage_name, subpackage)
else:
    import tensor_inpainting_agent  # noqa: E402,F401

from tensor_inpainting_agent.observability.trace_logger import TraceLogger
from tensor_inpainting_agent.core.llm import TensorInpaintingLLM
from tensor_inpainting_agent.tools.base import Tool, ToolParameter
from tensor_inpainting_agent.tools.errors import ToolErrorCode
from tensor_inpainting_agent.tools.registry import ToolRegistry
from tensor_inpainting_agent.tools.response import ToolResponse, ToolStatus

__all__ = [
    "Tool",
    "ToolParameter",
    "ToolErrorCode",
    "ToolRegistry",
    "ToolResponse",
    "ToolStatus",
    "TraceLogger",
    "TensorInpaintingLLM",
]
