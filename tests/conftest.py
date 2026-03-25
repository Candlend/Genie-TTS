"""Pytest configuration and shared fixtures.

Stubs out heavy runtime dependencies so that lightweight utility modules
can be imported and tested without the full inference stack installed.
"""
import sys
import os
from unittest.mock import MagicMock

# Ensure src/ is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Tell Genie not to check for model dirs at import time
os.environ.setdefault("GENIE_SKIP_RESOURCE_CHECK", "1")


class _ModuleStub(MagicMock):
    """A MagicMock that also supports attribute access as sub-module stubs."""
    def __getattr__(self, name):
        child = _ModuleStub()
        object.__setattr__(self, name, child)
        # Register as a sys.modules entry so `from x.y import z` works
        parent_name = self.__class__.__name__
        return child


def _stub(name: str):
    """Install a stub module for *name* and all its potential sub-paths."""
    if name not in sys.modules:
        mock = MagicMock()
        sys.modules[name] = mock
        # Also register common sub-modules so `from pkg.sub import X` works
        return mock
    return sys.modules[name]


# ---- Stub heavy top-level packages ----
for _top in [
    "onnx", "onnxruntime",
    "sounddevice", "soundfile", "soxr",
    "tokenizers",
    "huggingface_hub",
    "pydantic",
    # Note: G2P deps (pypinyin, nltk, g2p_en, pyopenjtalk, g2pk2, jamo, etc.) are NOT stubbed
    # so that pytest.importorskip() in G2P tests can detect missing packages and skip correctly.
]:
    _stub(_top)

# ---- fastapi needs sub-module stubs for `from fastapi.responses import ...` ----
_fastapi = _stub("fastapi")
_fastapi_resp = MagicMock()
_fastapi_resp.StreamingResponse = MagicMock
_fastapi_resp.JSONResponse = MagicMock
for _sub in ["responses", "middleware", "middleware.cors", "routing"]:
    sys.modules[f"fastapi.{_sub}"] = _fastapi_resp

# ---- uvicorn ----
_uvicorn = _stub("uvicorn")
for _sub in ["config", "main"]:
    sys.modules[f"uvicorn.{_sub}"] = MagicMock()

# ---- yaml (pyyaml) ----
if "yaml" not in sys.modules:
    sys.modules["yaml"] = MagicMock()
