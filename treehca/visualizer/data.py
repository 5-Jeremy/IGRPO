"""Reconstruct forests without dropping or rewriting saved node fields."""

from __future__ import annotations

import colorsys
import hashlib
import json
import math
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any


@dataclass
class Node:
    uid: str
    parent: str | None
    depth: int
    records: list[dict[str, Any]] = field(default_factory=list)
    lines: list[int] = field(default_factory=list)
    children: list[str] = field(default_factory=list)


@dataclass
class Tree:
    uid: str
    grouping: str
    nodes: dict[str, Node]
    warnings: list[str]

    @property
    def statistics(self):
        return {
            "tree_id": self.uid,
            "grouping": self.grouping,
            "saved_nodes": sum(bool(n.records) for n in self.nodes.values()),
            "saved_records": sum(len(n.records) for n in self.nodes.values()),
            "leaf_count": sum(not n.children for n in self.nodes.values()),
            "maximum_depth": max(n.depth for n in self.nodes.values()),
            "missing_ancestors": sum(not n.records and n.uid != "root" for n in self.nodes.values()),
        }


@dataclass
class Document:
    trees: list[Tree]
    record_count: int
    warnings: list[str]


def discover_files(directory: Path) -> list[Path]:
    """Scan immediate JSONL children; natural ordering puts 2 before 10."""
    directory = directory.expanduser().resolve()
    if not directory.is_dir():
        raise ValueError(f"Not a rollout directory: {directory}")
    return sorted(
        (p for p in directory.glob("*.jsonl") if p.is_file() and p.resolve().parent == directory),
        key=lambda p: [(0, int(part)) if part.isdigit() else (1, part) for part in re.split(r"(\d+)", p.name)],
    )


def load_file(path: Path) -> Document:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError as exc:
                raise ValueError(f"{path.name}, line {line_number}: invalid JSON; file may still be writing. Refresh to retry.") from exc
            ancestry = record.get("node_path") if isinstance(record, dict) else None
            if not isinstance(ancestry, list) or len(ancestry) < 2 or not all(isinstance(x, str) and x for x in ancestry) or ancestry[-1] != "root" or len(set(ancestry)) != len(ancestry):
                raise ValueError(f"{path.name}, line {line_number}: expected a unique node-to-root node_path ending in 'root'.")
            rows.append((line_number, record))
    if not rows:
        raise ValueError(f"{path.name}: no rollout records.")

    # A top-level branch is always identifiable, even with missing ancestor rows.
    branches: dict[str, list[tuple[int, dict]]] = {}
    for line_number, record in rows:
        branches.setdefault(record["node_path"][-2], []).append((line_number, record))
    groups: dict[tuple, list[tuple[int, dict]]] = {}
    for branch_uid, branch_rows in branches.items():
        ids = {str(r.get("tree_uid", r.get("uid"))) for _, r in branch_rows if r.get("tree_uid", r.get("uid")) is not None}
        if len(ids) > 1:
            raise ValueError(f"{path.name}: conflicting tree IDs in branch {branch_uid}.")
        root_inputs = {r["input"] for _, r in branch_rows if len(r["node_path"]) == 2 and isinstance(r.get("input"), str)}
        if ids:
            key = ("saved tree ID", next(iter(ids)))
        elif len(root_inputs) == 1:
            key = ("inferred from identical initial input", next(iter(root_inputs)))
        else:
            key = ("unresolved initial branch", branch_uid)
        groups.setdefault(key, []).extend(branch_rows)

    trees = []
    for index, ((grouping, group_id), group_rows) in enumerate(groups.items(), 1):
        nodes = {"root": Node("root", None, 0)}
        warnings = []
        if grouping != "saved tree ID":
            warnings.append("Tree grouping is inferred: identical initial inputs may merge separate episodes. Save uid/tree_uid to make grouping exact." if grouping.startswith("inferred") else "Initial input is unavailable or ambiguous; this tree shows one initial branch only.")
        for line_number, record in sorted(group_rows):
            chain = list(reversed(record["node_path"]))
            for depth, uid in enumerate(chain[1:], 1):
                parent = chain[depth - 1]
                if uid in nodes and (nodes[uid].parent != parent or nodes[uid].depth != depth):
                    raise ValueError(f"{path.name}, line {line_number}: conflicting ancestry for {uid}.")
                if uid not in nodes:
                    nodes[uid] = Node(uid, parent, depth)
                    nodes[parent].children.append(uid)
            node = nodes[record["node_path"][0]]
            node.records.append(record)
            node.lines.append(line_number)
        missing = sum(not n.records and n.uid != "root" for n in nodes.values())
        duplicates = sum(len(n.records) > 1 for n in nodes.values())
        if missing:
            warnings.append(f"{missing} ancestor nodes have no saved record; shown as placeholders.")
        if duplicates:
            warnings.append(f"{duplicates} nodes have multiple saved records; select a record in Node details to inspect every copy.")
        trees.append(Tree(group_id if grouping == "saved tree ID" else f"inferred-{index}", grouping, nodes, warnings))
    return Document(trees, len(rows), [])


class FileCache:
    """Bound memory while automatically reloading changed files."""

    def __init__(self, directory: Path, capacity: int = 3):
        self.directory = directory.expanduser().resolve()
        self.capacity = capacity
        self.entries = OrderedDict()
        self.lock = RLock()

    def load(self, filename: str) -> Document:
        # Only expose files from the configured directory, including on callback requests.
        if filename not in {p.name for p in discover_files(self.directory)}:
            raise ValueError(f"Unknown rollout file: {filename}")
        path = self.directory / filename
        with self.lock:
            stat = path.stat()
            stamp = (stat.st_mtime_ns, stat.st_size)
            if filename not in self.entries or self.entries[filename][0] != stamp:
                document = load_file(path)
                self.entries[filename] = (stamp, document)
            self.entries.move_to_end(filename)
            while len(self.entries) > self.capacity:
                self.entries.popitem(last=False)
            return self.entries[filename][1]


def numeric(value):
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (ValueError, TypeError, OverflowError):
        return None


def page_type_style(node: Node):
    """Stable categorical colors across files, including the native empty index name."""
    if node.uid == "root":
        return "Synthetic root", "#f1f5f9"
    page = node.records[0].get("page_type") if node.records else None
    if page is None:
        return "Missing page type", "#cbd5e1"
    page = str(page)
    colors = {"": "#93c5fd", "index": "#93c5fd", "search_results": "#fcd34d", "item_page": "#6ee7b7", "item_sub_page": "#c4b5fd", "done": "#fda4af"}
    if page in colors:
        return 'Initial search ("")' if page == "" else page, colors[page]
    # Future page categories keep the same color without depending on discovery order.
    hue = int.from_bytes(hashlib.sha256(page.encode()).digest()[:2], "big") / 65536
    color = "#" + "".join(f"{round(channel * 255):02x}" for channel in colorsys.hls_to_rgb(hue, 0.78, 0.65))
    return page, color


def graph_elements(tree: Tree, metric: str | None = None):
    """Fixed depth rows and centered parents, matching the reference viewer."""
    positions = {}
    column = 0
    # Iterative postorder also handles long paths without Python recursion limits.
    stack = [("root", False)]
    while stack:
        uid, visited = stack.pop()
        node = tree.nodes[uid]
        if node.children and not visited:
            stack.append((uid, True))
            stack.extend((child, False) for child in reversed(node.children))
            continue
        if node.children:
            x = (positions[node.children[0]]["x"] + positions[node.children[-1]]["x"]) / 2
        else:
            x = column * 210
            column += 1
        positions[uid] = {"x": x, "y": node.depth * 110}
    values = {uid: numeric(n.records[0].get(metric)) if n.records and metric else None for uid, n in tree.nodes.items()}
    finite = [v for v in values.values() if v is not None]
    limits = (min(finite), max(finite)) if finite else None
    elements = []
    for uid, node in tree.nodes.items():
        kind = "root" if uid == "root" else ("internal" if node.children else "leaf")
        color = {"root": "#56b4e9", "internal": "#e69f00", "leaf": "#009e73"}[kind]
        foreground = "#0f172a"
        if metric == "page_type":
            _, color = page_type_style(node)
        elif metric:
            value = values[uid]
            color = "#e2e8f0"
            if value is not None and limits:
                low, high = limits
                ratio = 0.5 if low == high else (value - low) / (high - low)
                color = "#" + "".join(f"{round(a + ratio * (b - a)):02x}" for a, b in zip((224, 242, 254), (8, 47, 73)))
                foreground = "white" if ratio > 0.55 else "#0f172a"
        output = str(node.records[0].get("output", "")) if node.records else ""
        action = re.search(r"<action>(.*?)</action>", output, re.DOTALL)
        label = action.group(1).strip() if action else uid
        if uid == "root":
            label = "root (synthetic)"
        elif not node.records:
            label = f"missing: {uid}"
        elements.append({"data": {"id": uid, "label": label, "color": color, "foreground": foreground}, "position": positions[uid], "classes": kind, "grabbable": False})
        if node.parent is not None:
            elements.append({"data": {"id": f"edge:{tree.uid}:{uid}", "source": node.parent, "target": uid}})
    return elements, limits
