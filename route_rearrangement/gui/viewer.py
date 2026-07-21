"""PyQt5 viewer: the original literature route above, one rearrangement below, Prev/Next.

Borrows the rendering core from ``synthesis_extraction.gui`` (via :mod:`.render`) and the
single-route display idiom from its ``PathwayViewerDialog`` (a scrollable scheme image),
extended to show two schemes stacked — original vs current rearrangement — plus a metrics
panel and a sort-by-metric selector so you can walk the rearrangements best-first and check
each is chemically sensible.

Launch via ``python -m route_rearrangement.gui`` (see :mod:`.__main__`).
"""

from __future__ import annotations

import os
import shutil
import tempfile
from functools import lru_cache
from typing import Dict, List, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QComboBox, QFileDialog, QHBoxLayout, QLabel, QMainWindow, QMessageBox,
    QPushButton, QScrollArea, QSplitter, QVBoxLayout, QWidget)

from ..metrics import METRIC_NAMES
from .model import DISTINCT_KEY, RouteEntry, TreeGroup
from .render import render_record_png

# resolution of the downloaded PNGs — higher than the on-screen preview
EXPORT_DPI = 220


class RouteViewer(QMainWindow):
    def __init__(self, group: TreeGroup, sort_metric: Optional[str] = None, dpi: int = 130):
        super().__init__()
        self.group = group
        self.dpi = dpi
        self._work = tempfile.mkdtemp(prefix="route_gui_")
        self._idx = 0
        keys = group.sort_keys()                    # "distinct" first when computed
        self._sort = sort_metric or (keys[0] if keys else None)
        self._order: List[RouteEntry] = group.sorted_rearrangements(self._sort)

        self.setWindowTitle(f"Route rearrangements — {group.tree_id}")
        self.resize(1200, 900)
        self._build_ui(keys)
        self._show_original()
        self._show_current()

    # -- UI construction ---------------------------------------------------------------
    def _build_ui(self, keys: List[str]) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # human labels for the sort dropdown (the dissimilarity key gets a descriptive name)
        self._sort_labels = {DISTINCT_KEY: "most different (dissimilarity)"}
        bar = QHBoxLayout()
        bar.addWidget(QLabel("<b>%s</b>" % self.group.tree_id))
        bar.addWidget(QLabel("   sort rearrangements by:"))
        self.sort_box = QComboBox()
        self._label_to_key = {self._sort_labels.get(k, k): k for k in keys}
        display = [self._sort_labels.get(k, k) for k in keys] or ["(none)"]
        self.sort_box.addItems(display)
        if self._sort in keys:
            self.sort_box.setCurrentText(self._sort_labels.get(self._sort, self._sort))
        self.sort_box.currentTextChanged.connect(self._resort)
        bar.addWidget(self.sort_box)
        self.save_orig_btn = QPushButton("⤓ Save literature PNG")
        self.save_cur_btn = QPushButton("⤓ Save this rearrangement PNG")
        self.save_both_btn = QPushButton("⤓ Save both…")
        self.save_orig_btn.clicked.connect(self._save_original)
        self.save_cur_btn.clicked.connect(self._save_current)
        self.save_both_btn.clicked.connect(self._save_both)
        bar.addWidget(self.save_orig_btn)
        bar.addWidget(self.save_cur_btn)
        bar.addWidget(self.save_both_btn)
        self.prev_btn = QPushButton("◀ Prev")
        self.next_btn = QPushButton("Next ▶")
        self.prev_btn.clicked.connect(self._prev)
        self.next_btn.clicked.connect(self._next)
        bar.addStretch(1)
        bar.addWidget(self.prev_btn)
        bar.addWidget(self.next_btn)
        root.addLayout(bar)

        split = QSplitter(Qt.Vertical)
        root.addWidget(split, 1)

        self.orig_label = QLabel(alignment=Qt.AlignCenter)
        self.orig_caption = QLabel(alignment=Qt.AlignLeft)
        self.cur_label = QLabel(alignment=Qt.AlignCenter)
        self.cur_caption = QLabel(alignment=Qt.AlignLeft)
        split.addWidget(self._pane("Original literature route", self.orig_caption, self.orig_label))
        split.addWidget(self._pane("Rearrangement", self.cur_caption, self.cur_label))
        split.setSizes([420, 480])

    def _pane(self, title: str, caption: QLabel, image: QLabel) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        head = QLabel(f"<b>{title}</b>")
        caption.setTextFormat(Qt.RichText)
        caption.setWordWrap(True)
        scroll = QScrollArea()
        scroll.setWidgetResizable(False)
        scroll.setWidget(image)
        lay.addWidget(head)
        lay.addWidget(caption)
        lay.addWidget(scroll, 1)
        return w

    # -- rendering ---------------------------------------------------------------------
    @lru_cache(maxsize=256)
    def _png(self, ordering_key: str) -> Optional[str]:
        entry = self._entry_by_key[ordering_key]
        try:
            return render_record_png(entry.record, self._work, dpi=self.dpi)
        except Exception:
            return None

    def _audit_line(self, entry: RouteEntry) -> str:
        """Post-hoc findings, if a feasibility.jsonl was joined in.  A check the route's own
        literature ordering also trips is inherent chemistry, not rearrangement damage, so
        the two are counted separately."""
        if not entry.has_audit():
            return ""
        findings = entry.findings()
        if not findings:
            return "<br><span style='color:#2f855a'>✓ feasibility audit: no findings</span>"
        new = set(self.group.new_checks(entry)) if not entry.is_original else set()
        detail = ", ".join(
            f"{c}{'*' if c in new else ''}" for c in sorted(entry.checks()))
        counts = f"{entry.n_infeasible()} infeasible / {entry.n_risk()} risk"
        star = f" &nbsp; <b>{len(new)} new vs literature (*)</b>" if new else ""
        return (f"<br><span style='color:#b7791f'>audit: {counts} &nbsp; {detail}"
                f"</span>{star}")

    def _caption(self, entry: RouteEntry, extra: str = "") -> str:
        ordering = " → ".join(str(s) for s in entry.ordering)
        cells = []
        for m in METRIC_NAMES:
            v = entry.score(m)
            if v is None:
                continue
            base = self.group.original.score(m) if self.group.original is not None else None
            delta = ""
            if base is not None and not entry.is_original:
                d = v - base
                colour = "#2f855a" if d > 0 else ("#c53030" if d < 0 else "#888")
                delta = f" <span style='color:{colour}'>({d:+.2f})</span>"
            pct = self.group.percentile(entry, m)
            pct_s = f" [{pct:.0%}]" if pct is not None else ""
            cells.append(f"{m}={v:.3f}{delta}{pct_s}")
        metric_line = " &nbsp; ".join(cells) if cells else "no metrics computed"
        dist = entry.distance_to_original()
        dist_line = ""
        if dist is not None and not entry.is_original:
            dr = entry.diverse_rank()
            badge = (f"<b style='color:#2b6cb0'>most-different #{dr}</b> &nbsp; " if dr else "")
            dist_line = (f"<br><span style='color:#2b6cb0'>{badge}"
                         f"distance from literature route: {dist:.2f}</span>")
        pin = ("<b style='color:#805ad5'>[pinned]</b> " if entry.pinned else "")
        return (f"{pin}<span style='font-family:monospace'>order: {ordering}</span>{extra}<br>"
                f"<span style='color:#444'>{metric_line}</span>{dist_line}"
                f"{self._audit_line(entry)}")

    def _set_image(self, label: QLabel, entry: RouteEntry) -> None:
        self._entry_by_key = getattr(self, "_entry_by_key", {})
        key = ",".join(map(str, entry.ordering))
        self._entry_by_key[key] = entry
        png = self._png(key)
        if png:
            label.setPixmap(QPixmap(png))
            label.adjustSize()
        else:
            label.setText("render failed")

    def _show_original(self) -> None:
        if self.group.original is None:
            self.orig_caption.setText("(original ordering not present in this file)")
            return
        self.orig_caption.setText(self._caption(self.group.original))
        self._set_image(self.orig_label, self.group.original)

    def _show_current(self) -> None:
        if not self._order:
            self.cur_caption.setText("(no rearrangements)")
            return
        self._idx %= len(self._order)
        entry = self._order[self._idx]
        sort_name = self._sort_labels.get(self._sort, self._sort)
        rank = f" &nbsp; <b>#{self._idx + 1} of {len(self._order)}</b> by {sort_name}"
        self.cur_caption.setText(self._caption(entry, extra=rank))
        self._set_image(self.cur_label, entry)

    # -- navigation --------------------------------------------------------------------
    def _prev(self) -> None:
        self._idx -= 1
        self._show_current()

    def _next(self) -> None:
        self._idx += 1
        self._show_current()

    def _resort(self, label: str) -> None:
        key = self._label_to_key.get(label, label)
        if key in METRIC_NAMES or key == DISTINCT_KEY:
            self._sort = key
            self._order = self.group.sorted_rearrangements(key)
            self._idx = 0
            self._show_original()
            self._show_current()

    # -- image download ----------------------------------------------------------------
    def _current_entry(self) -> Optional[RouteEntry]:
        if not self._order:
            return None
        return self._order[self._idx % len(self._order)]

    def _export_png(self, entry: RouteEntry, dest: str) -> bool:
        """Render *entry* at export resolution and copy the PNG to *dest*."""
        try:
            png = render_record_png(entry.record, self._work, dpi=EXPORT_DPI)
            shutil.copyfile(png, dest)
            return True
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Save failed", f"Could not save image:\n{exc}")
            return False

    def _default_name(self, entry: RouteEntry, kind: str) -> str:
        order = "-".join(map(str, entry.ordering))
        return f"{self.group.tree_id}_{kind}_{order}.png"

    def _save_entry(self, entry: Optional[RouteEntry], kind: str) -> None:
        if entry is None:
            QMessageBox.information(self, "Nothing to save", "No route is displayed.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, f"Save {kind} route image", self._default_name(entry, kind),
            "PNG image (*.png)")
        if path:
            if not path.lower().endswith(".png"):
                path += ".png"
            if self._export_png(entry, path):
                self.statusBar().showMessage(f"Saved {os.path.basename(path)}", 5000)

    def _save_original(self) -> None:
        self._save_entry(self.group.original, "literature")

    def _save_current(self) -> None:
        self._save_entry(self._current_entry(), "rearrangement")

    def _save_both(self) -> None:
        """Save the literature route and the current rearrangement into a chosen folder."""
        entry = self._current_entry()
        folder = QFileDialog.getExistingDirectory(self, "Choose a folder to save both PNGs")
        if not folder:
            return
        saved = []
        if self.group.original is not None:
            dest = os.path.join(folder, self._default_name(self.group.original, "literature"))
            if self._export_png(self.group.original, dest):
                saved.append(os.path.basename(dest))
        if entry is not None:
            dest = os.path.join(folder, self._default_name(entry, "rearrangement"))
            if self._export_png(entry, dest):
                saved.append(os.path.basename(dest))
        if saved:
            self.statusBar().showMessage("Saved " + ", ".join(saved), 6000)


def launch(group: TreeGroup, sort_metric: Optional[str] = None, dpi: int = 130) -> int:
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv)
    viewer = RouteViewer(group, sort_metric=sort_metric, dpi=dpi)
    viewer.show()
    return app.exec_()
