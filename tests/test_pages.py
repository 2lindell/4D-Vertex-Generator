"""The committed GitHub Pages site (docs/) must match the current app and package."""
from __future__ import annotations

import filecmp
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_docs_site_is_up_to_date(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location("build_pages", ROOT / "tools" / "build_pages.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fresh = tmp_path / "site"
    module.build(fresh)
    docs = ROOT / "docs"
    built = sorted(p.relative_to(fresh) for p in fresh.rglob("*") if p.is_file())
    committed = sorted(p.relative_to(docs) for p in docs.rglob("*") if p.is_file())
    assert built == committed, "docs/ has missing or extra files: run python tools/build_pages.py"
    stale = [str(p) for p in built if not filecmp.cmp(fresh / p, docs / p, shallow=False)]
    assert not stale, f"docs/ is out of date ({', '.join(stale)}): run python tools/build_pages.py"
