"""检查后端包边界和禁止的反向依赖。"""

import ast
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[1] / "app"


@pytest.mark.parametrize("package", ["api/v1", "consultation", "knowledge", "infrastructure", "models"])
def test_core_packages_have_clear_locations(package: str) -> None:
    assert (APP / package / "__init__.py").is_file()


@pytest.mark.parametrize("package", ["consultation", "knowledge", "infrastructure", "models"])
def test_non_transport_packages_do_not_import_api(package: str) -> None:
    for path in (APP / package).rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        imports += [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
        assert not any(name and (name == "app.api" or name.startswith("app.api.")) for name in imports), path


def test_knowledge_does_not_depend_on_consultation() -> None:
    for path in (APP / "knowledge").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        imports += [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
        assert not any(
            name and (name == "app.consultation" or name.startswith("app.consultation.")) for name in imports
        ), path
