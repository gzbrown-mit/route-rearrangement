"""Static HTML gallery of one literature route and its enumerated rearrangements.

A display-free alternative to the PyQt viewer (works over SSH / without a display): renders
every route to a PNG via the borrowed synthesis_extraction renderer, embeds them inline,
and shows the metric scores with the original's value highlighted.  Rearrangements are
ordered best-first by a chosen metric.
"""

from __future__ import annotations

import base64
import html
import os
from typing import Optional

from ..metrics import METRIC_NAMES
from .model import RouteEntry, TreeGroup, load_groups
from .render import render_record_png


def _img_data_uri(png_path: str) -> str:
    with open(png_path, "rb") as fh:
        return "data:image/png;base64," + base64.b64encode(fh.read()).decode("ascii")


def _metric_cells(group: TreeGroup, entry: RouteEntry) -> str:
    cells = []
    for m in METRIC_NAMES:
        v = entry.score(m)
        if v is None:
            cells.append('<td class="na">–</td>')
            continue
        pct = group.percentile(entry, m)
        pct_s = f"<br><small>pctile {pct:.0%}</small>" if pct is not None else ""
        cells.append(f"<td>{v:.3f}{pct_s}</td>")
    return "".join(cells)


def _route_block(group: TreeGroup, entry: RouteEntry, work_dir: str,
                 title: str, dpi: int) -> str:
    download = ""
    try:
        png = render_record_png(entry.record, work_dir, dpi=dpi)
        uri = _img_data_uri(png)
        kind = "literature" if entry.is_original else "rearrangement"
        fname = f"{group.tree_id}_{kind}_{'-'.join(map(str, entry.ordering))}.png"
        # the image itself is a download link, plus an explicit button
        img = f'<a href="{uri}" download="{html.escape(fname)}"><img src="{uri}" alt="route"></a>'
        download = (f'<a class="dl" href="{uri}" download="{html.escape(fname)}">'
                    f'⤓ download PNG</a>')
    except Exception as e:  # noqa: BLE001
        img = f'<p class="err">render failed: {html.escape(str(e))}</p>'
    ordering = " → ".join(str(s) for s in entry.ordering)
    flags = entry.record.get("flags", {})
    fg_risk = len(flags.get("fg_risk", []) or [])
    flag_note = f' · <span class="flag">fg_risk: {fg_risk}</span>' if fg_risk else ""
    dist = entry.distance_to_original()
    dr = entry.diverse_rank()
    dist_note = ""
    if dist is not None and not entry.is_original:
        badge = f'<span class="diverse">most-different #{dr}</span> ' if dr else ""
        dist_note = f' · {badge}<span class="dist">distance from literature: {dist:.2f}</span>'
    return f"""
    <section class="route {'original' if entry.is_original else ''}">
      <h3>{html.escape(title)} {download}</h3>
      <p class="ordering">ordering (first→last): {html.escape(ordering)}{flag_note}{dist_note}</p>
      <table class="metrics"><tr>{''.join(f'<th>{m}</th>' for m in METRIC_NAMES)}</tr>
      <tr>{_metric_cells(group, entry)}</tr></table>
      <div class="scheme">{img}</div>
    </section>"""


def build_gallery(group: TreeGroup, out_html: str, *, work_dir: str,
                  sort_metric: Optional[str] = None, top: int = 25, dpi: int = 130) -> str:
    os.makedirs(work_dir, exist_ok=True)
    keys = group.sort_keys()                       # "distinct" first when computed
    sort_metric = sort_metric or (keys[0] if keys else None)

    from .model import DISTINCT_KEY
    if sort_metric == DISTINCT_KEY:
        heading = (f'Rearrangements (most different from the literature route and each other '
                   f'first; top {top})')
    else:
        heading = f'Rearrangements (best-first by {html.escape(sort_metric or "n/a")}, top {top})'

    parts = [_HEAD.format(tree_id=html.escape(group.tree_id),
                          sort=html.escape("dissimilarity (most different first)"
                                           if sort_metric == DISTINCT_KEY else
                                           (sort_metric or "file order")),
                          n=len(group.rearrangements))]
    if group.original is not None:
        parts.append('<h2>Original literature route</h2>')
        parts.append(_route_block(group, group.original, work_dir, "literature ordering", dpi))
    parts.append(f'<h2>{heading}</h2>')
    for i, entry in enumerate(group.sorted_rearrangements(sort_metric)[:top], 1):
        parts.append(_route_block(group, entry, work_dir, f"rearrangement #{i}", dpi))
    parts.append("</body></html>")
    with open(out_html, "w") as fh:
        fh.write("\n".join(parts))
    return out_html


_HEAD = """<!doctype html><html><head><meta charset="utf-8">
<title>Route rearrangements — {tree_id}</title>
<style>
 body {{ font-family: -apple-system, Helvetica, Arial, sans-serif; margin: 24px; color: #1a1a1a; }}
 h1 {{ font-size: 20px; }} h2 {{ margin-top: 28px; border-bottom: 2px solid #eee; }}
 section.route {{ border: 1px solid #ddd; border-radius: 8px; padding: 12px 16px; margin: 14px 0; }}
 section.original {{ border-color: #2b6cb0; background: #f5f9ff; }}
 .ordering {{ font-family: ui-monospace, Menlo, monospace; font-size: 12px; color: #555; }}
 table.metrics {{ border-collapse: collapse; margin: 8px 0; }}
 table.metrics th, table.metrics td {{ border: 1px solid #ddd; padding: 4px 10px; font-size: 12px;
   text-align: center; }}
 table.metrics th {{ background: #f3f3f3; }} td.na {{ color: #bbb; }}
 .scheme {{ overflow-x: auto; }} .scheme img {{ max-width: none; height: auto; }}
 .flag {{ color: #b7791f; }} .err {{ color: #c53030; }}
 .dist {{ color: #2b6cb0; }}
 .diverse {{ background: #2b6cb0; color: #fff; border-radius: 4px; padding: 1px 6px;
   font-weight: 600; }}
 a.dl {{ font-size: 12px; font-weight: normal; color: #2b6cb0; text-decoration: none;
   margin-left: 10px; }}
 a.dl:hover {{ text-decoration: underline; }}
 small {{ color: #888; }}
</style></head><body>
<h1>Route rearrangements for {tree_id}</h1>
<p>{n} valid rearrangements enumerated. Sorted by <b>{sort}</b>. Higher metric score = better;
percentile = the route's standing among all rearrangements of this literature pathway.</p>"""


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--routes", default="results/scored.jsonl")
    ap.add_argument("--tree-id", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--sort", default="", help="metric to sort by (default: first available)")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--dpi", type=int, default=130)
    args = ap.parse_args(argv)

    groups = load_groups(args.routes)
    if args.tree_id not in groups:
        ap.error(f"{args.tree_id} not in {args.routes}; have {sorted(groups)[:10]}...")
    group = groups[args.tree_id]
    out = args.out or f"results/gallery_{args.tree_id}.html"
    work = os.path.join(os.path.dirname(out) or ".", f"_imgs_{args.tree_id}")
    build_gallery(group, out, work_dir=work, sort_metric=args.sort or None,
                  top=args.top, dpi=args.dpi)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
