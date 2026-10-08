import importlib

import pytest

import qts

SUBPACKAGES = [
    "core",
    "data",
    "strategy",
    "risk",
    "oms",
    "portfolio",
    "costs",
    "broker",
    "recon",
    "safety",
    "persistence",
    "config",
    "secrets",
    "health",
    "research",
]


def test_package_imports() -> None:
    assert qts.ENGINE_VERSION == "0.1.0"


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_imports(name: str) -> None:
    assert importlib.import_module(f"qts.{name}") is not None
