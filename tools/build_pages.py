"""Build a static copy of the Streamlit app for GitHub Pages.

    python tools/build_pages.py [output_dir]      # default: docs

The app runs entirely in the visitor's browser with stlite (Streamlit on Pyodide, Python compiled to
WebAssembly), so the page needs no server and stays available. The output is index.html plus app.py
and the four_d_vertex_generator package, which index.html mounts into the in-browser file system.

GitHub Pages serves the committed docs/ folder ("Deploy from a branch", folder /docs). Rebuild and
commit docs/ after changing the app; tests/test_pages.py fails while docs/ is out of date.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STLITE_VERSION = "1.9.2"  # @stlite/browser on npm; bump deliberately and check the page still loads
REQUIREMENTS = ["numpy", "scipy", "pandas", "plotly"]

PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>4D Vertex Generator</title>
  <link rel="icon" type="image/svg+xml" href="assets/favicon.svg">
  <meta name="description"
        content="Generate 4D symmetry orbits and export them as 4OFF, in the browser.">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@stlite/browser@__VERSION__/build/stlite.css">
  <style>
    :root { color-scheme: light dark; }
    body { margin: 0; font-family: system-ui, sans-serif; background: Canvas; color: CanvasText; }
    #loading { max-width: 40rem; margin: 18vh auto 0; padding: 0 16px; line-height: 1.5; }
    #loading small { opacity: 0.7; }
  </style>
</head>
<body>
  <div id="root">
    <div id="loading">
      <h1>4D Vertex Generator</h1>
      <p>Loading Python in your browser. The first visit downloads about 30 MB (Python,
      numpy, scipy, pandas, plotly); later visits use the browser's cache and start much
      faster.</p>
      <p><small>Everything runs on your own computer; nothing is sent to a server. Large
      symmetry groups take longer here than in the locally installed app.</small></p>
      <p><small>Source, command-line tools and install instructions:
      <a href="https://github.com/2lindell/4D-Vertex-Generator">github.com/2lindell/4D-Vertex-Generator</a>.
      </small></p>
      <noscript><p>This page needs JavaScript.</p></noscript>
    </div>
  </div>
  <script type="module">
    import { mount } from "https://cdn.jsdelivr.net/npm/@stlite/browser@__VERSION__/build/stlite.js";
    // each file's address carries a hash of its contents, so a redeploy is never hidden by a cache
    const files = __FILES__;
    mount(
      {
        entrypoint: "app.py",
        requirements: __REQUIREMENTS__,
        files: Object.fromEntries(
          files.map(([path, hash]) => [path, { url: `./${path}?v=${hash}` }]),
        ),
      },
      document.getElementById("root"),
    );
  </script>
</body>
</html>
"""


def build(out: Path) -> list[str]:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    files = ["app.py", "assets/favicon.svg"]  # the icon is mounted too: app.py's page_icon reads it
    shutil.copy2(ROOT / "app.py", out / "app.py")
    (out / "assets").mkdir()
    shutil.copy2(ROOT / "assets" / "favicon.svg", out / "assets" / "favicon.svg")
    package = ROOT / "src" / "four_d_vertex_generator"
    for source in sorted(package.glob("*.py")):
        rel = f"four_d_vertex_generator/{source.name}"
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, out / rel)
        files.append(rel)
    hashed = [[rel, hashlib.sha256((out / rel).read_bytes()).hexdigest()[:12]] for rel in files]
    page = (
        PAGE.replace("__VERSION__", STLITE_VERSION)
        .replace("__FILES__", json.dumps(hashed))
        .replace("__REQUIREMENTS__", json.dumps(REQUIREMENTS))
    )
    (out / "index.html").write_text(page)
    (out / ".nojekyll").write_text("")  # serve files as they are
    return files


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs"
    mounted = build(target)
    print(f"wrote {target} ({len(mounted)} files mounted: {', '.join(mounted)})")
