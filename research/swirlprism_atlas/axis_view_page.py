"""The axis view page (assets/symmetry_axis.html) from the 3-sphere map's axis_view.json.

    python axis_view_page.py <sphere_map data dir> ../../assets/symmetry_axis.html
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

TEMPLATE = Path(__file__).with_name("axis_view_template.html")


def main(src_dir, out_path):
    data = json.loads((Path(src_dir) / "axis_view.json").read_text())
    # the key's names and colours for the shapes shown (from the domain page's own data)
    page = TEMPLATE.read_text().replace("__DATA__", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))
    Path(out_path).write_text(page)
    print(f"{out_path}: {len(page) / 1e6:.2f} MB; shapes {', '.join(s['id'] for s in data['shapes'])}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
