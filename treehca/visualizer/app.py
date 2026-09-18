"""Dash interface for browsing JSONL forests and exact saved records."""

from __future__ import annotations

import json
import re
from pathlib import Path

from .data import FileCache, discover_files, graph_elements, numeric

GRAPH_STYLE = [
    {
        "selector": "node",
        "style": {
            "label": "data(label)",
            "font-family": "ui-monospace, monospace",
            "font-size": 11,
            "text-wrap": "ellipsis",
            "text-max-width": 150,
            "text-valign": "center",
            "width": 170,
            "height": 48,
            "shape": "round-rectangle",
            "border-width": 2,
            "border-color": "#334155",
            "background-color": "data(color)",
            "color": "data(foreground)",
        },
    },
    {"selector": "node:selected", "style": {"border-width": 6, "border-color": "#111827"}},
    {"selector": "edge", "style": {"curve-style": "bezier", "width": 3, "line-color": "#475569", "target-arrow-color": "#0f172a", "target-arrow-shape": "triangle", "target-distance-from-node": 4, "arrow-scale": 1.6}},
]


def format_value(value):
    # JSON formatting retains nested structures, nulls, and full numeric precision.
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)


def create_app(log_dir: Path):
    import dash_cytoscape as cyto
    from dash import Dash, Input, Output, State, ctx, dcc, html

    cache = FileCache(log_dir)
    files = discover_files(log_dir)
    app = Dash(__name__, assets_folder=str(Path(__file__).with_name("assets")), title="WebShop Rollout Trees")

    def table(values):
        return html.Table(html.Tbody([html.Tr([html.Th(key), html.Td(html.Pre(format_value(value)))]) for key, value in values.items()]), className="value-table details-table")

    def get_tree(view):
        if not view:
            return None
        try:
            document = cache.load(view["file"])
            return document.trees[view["tree"]]
        except (ValueError, OSError, UnicodeError, IndexError):
            # A live writer can replace a file between dependent callbacks.
            return None

    app.layout = html.Div(
        [
            dcc.Store(id="view"),
            dcc.Store(id="selection"),
            dcc.Store(id="search-state"),
            dcc.Store(id="scroll-request"),
            dcc.Store(id="scroll-result"),
            html.Header(
                [
                    html.Div([html.H1("WebShop Rollout Trees"), html.Div(str(cache.directory), className="log-path")], className="title-block"),
                    html.Label(["JSONL file", dcc.Dropdown(id="file", options=[p.name for p in files], value=files[-1].name if files else None, clearable=False)], className="selector"),
                    html.Label(["Tree", dcc.Dropdown(id="tree", clearable=False)], className="selector"),
                    html.Button("Refresh files", id="refresh", className="fit-button"),
                    html.Button("Fit graph", id="fit", className="fit-button"),
                ],
                className="toolbar",
            ),
            html.Div(id="error", className="error-panel", role="alert"),
            html.Div(id="warning", className="warning-panel"),
            html.Main(
                [
                    html.Section(
                        [
                            html.Div(
                                [
                                    html.Div(id="legend", className="legend"),
                                    html.Label(["Node color", dcc.Dropdown(id="metric", options=[{"label": "Structural type", "value": ""}], value="", clearable=False)], className="color-mode-control"),
                                ],
                                className="graph-controls",
                            ),
                            cyto.Cytoscape(id="graph", elements=[], layout={"name": "preset", "fit": True, "padding": 36}, stylesheet=GRAPH_STYLE, minZoom=0.01, maxZoom=4, autoungrabify=True, responsive=True, style={"width": "100%", "height": "100%"}, className="tree-graph"),
                        ],
                        className="graph-panel",
                    ),
                    html.Aside(
                        [
                            html.H2("Node details · all saved fields"),
                            dcc.Dropdown(id="record", options=[], placeholder="Saved record (JSONL line)"),
                            html.Div(id="details", className="node-details-scroll", tabIndex=0),
                        ],
                        className="details-panel",
                    ),
                ],
                className="content-grid",
            ),
            html.Section(
                dcc.Tabs(
                    [
                        dcc.Tab(
                            label="Saved text & context",
                            children=[
                                html.Div(
                                    [
                                        html.Div([html.H3("Initial input"), html.Pre(id="context", className="node-text")], className="tab-text-panel"),
                                        html.Div(
                                            [
                                                html.Div(
                                                    [
                                                        dcc.Dropdown(id="text-field", options=["input", "output", "Raw record"], value="output", clearable=False, style={"minWidth": "130px"}),
                                                        dcc.Input(id="query", type="search", placeholder="Search saved text…", debounce=0.25),
                                                        html.Button("Previous", id="previous"),
                                                        html.Button("Next", id="next"),
                                                        html.Button("Jump to end", id="end"),
                                                    ],
                                                    className="text-controls",
                                                ),
                                                html.Div([dcc.Checklist(id="case", options=["Match case"], value=[]), html.Span(id="search-count")], className="search-options"),
                                                html.Pre(id="node-text", className="node-text", tabIndex=0),
                                            ],
                                            className="tab-text-panel",
                                        ),
                                    ],
                                    className="context-tab-grid",
                                )
                            ],
                        ),
                        dcc.Tab(label="Tree statistics", children=html.Div(id="tree-stats", className="table-scroll")),
                        dcc.Tab(label="File statistics", children=html.Div(id="file-stats", className="table-scroll")),
                    ]
                ),
                className="summary-area",
            ),
        ],
        className="app-shell",
    )

    @app.callback(Output("file", "options"), Output("file", "value"), Input("refresh", "n_clicks"), State("file", "value"))
    def refresh_files(clicks, selected):
        names = [p.name for p in discover_files(log_dir)]
        return names, selected if selected in names else (names[-1] if names else None)

    @app.callback(Output("tree", "options"), Output("tree", "value"), Output("error", "children"), Output("file-stats", "children"), Input("file", "value"), Input("refresh", "n_clicks"))
    def select_file(filename, clicks):
        if not filename:
            return [], None, "No JSONL files found. Refresh after logs are saved.", []
        try:
            document = cache.load(filename)
            options = []
            for index, tree in enumerate(document.trees):
                first = next((n.records[0] for n in tree.nodes.values() if n.depth == 1 and n.records), {})
                prompt = first.get("input", "")
                task = re.search(r"Your task is to:\s*(.*)", prompt) if isinstance(prompt, str) else None
                label = task.group(1)[:110] if task else tree.uid
                options.append({"label": f"{index + 1} · {tree.statistics['saved_nodes']} nodes · {label}", "value": index})
            return options, 0, "", table({"file": filename, "trees": len(document.trees), "records": document.record_count})
        except (ValueError, OSError, UnicodeError) as exc:
            return [], None, str(exc), []

    @app.callback(Output("view", "data"), Output("warning", "children"), Output("tree-stats", "children"), Output("context", "children"), Output("metric", "options"), Input("file", "value"), Input("tree", "value"), Input("refresh", "n_clicks"))
    def select_tree(filename, index, clicks):
        empty = (None, "", [], "", [{"label": "Structural type", "value": ""}])
        if filename is None or index is None:
            return empty
        try:
            tree = cache.load(filename).trees[index]
        except (ValueError, OSError, UnicodeError, IndexError):
            return empty
        fields = sorted({key for node in tree.nodes.values() for record in node.records for key, value in record.items() if numeric(value) is not None})
        context = next((n.records[0].get("input", "") for n in tree.nodes.values() if n.depth == 1 and n.records), "Initial input was not saved.")
        return {"file": filename, "tree": index, "refresh": clicks}, " ".join(tree.warnings), table(tree.statistics), format_value(context), [{"label": "Structural type", "value": ""}] + [{"label": f, "value": f} for f in fields]

    @app.callback(Output("graph", "elements"), Output("graph", "layout"), Output("legend", "children"), Input("view", "data"), Input("metric", "value"), Input("fit", "n_clicks"))
    def render_graph(view, metric, clicks):
        tree = get_tree(view)
        elements, limits = graph_elements(tree, metric) if tree else ([], None)
        if metric:
            legend = f"{metric}: {limits[0]:.6g} → {limits[1]:.6g} · darker = higher; gray = missing/nonfinite" if limits else f"{metric}: no finite values"
            legend += " · first record per node"
        else:
            legend = [html.Span([html.I(className=f"legend-swatch {kind}-swatch"), label]) for kind, label in [("root", "Synthetic root"), ("internal", "Internal"), ("leaf", "Leaf")]]
        return elements, {"name": "preset", "fit": True, "padding": 36 + (clicks or 0) % 2}, legend

    @app.callback(Output("selection", "data"), Output("record", "options"), Output("record", "value"), Input("view", "data"), Input("graph", "tapNodeData"))
    def select_node(view, tapped):
        tree = get_tree(view)
        if tree is None:
            return None, [], None
        uid = tapped.get("id") if ctx.triggered_id == "graph" and tapped else "root"
        node = tree.nodes.get(uid, tree.nodes["root"])
        return {**view, "node": node.uid}, [{"label": f"Record {i + 1} · line {line}", "value": i} for i, line in enumerate(node.lines)], 0 if node.records else None

    @app.callback(
        Output("details", "children"),
        Output("node-text", "children"),
        Output("search-count", "children"),
        Output("search-state", "data"),
        Output("scroll-request", "data"),
        Input("selection", "data"),
        Input("record", "value"),
        Input("text-field", "value"),
        Input("query", "value"),
        Input("case", "value"),
        Input("previous", "n_clicks"),
        Input("next", "n_clicks"),
        Input("query", "n_submit"),
        Input("end", "n_clicks"),
        State("search-state", "data"),
    )
    def show_record(selection, record_index, text_field, query, case, previous, following, submit, end, search):
        tree = get_tree(selection)
        if tree is None or selection["node"] not in tree.nodes:
            return [], "", "", {}, {"kind": "reset"}
        node = tree.nodes[selection["node"]]
        record_index = min(record_index or 0, max(0, len(node.records) - 1))
        record = node.records[record_index] if node.records else {}
        derived = {"node_uid (derived)": node.uid, "parent (derived)": node.parent, "graph depth": node.depth, "children (derived)": node.children, "JSONL lines": node.lines}
        details = [html.H3("Structure"), table(derived), html.H3("Saved fields"), table(record)]
        if not node.records:
            details.append(html.P("Synthetic root; no record was logged." if node.uid == "root" else "Missing ancestor; no record was logged."))
        text = json.dumps(record, ensure_ascii=False, indent=2) if text_field == "Raw record" else format_value(record[text_field]) if text_field in record else ""
        matches = list(re.finditer(re.escape(query), text, 0 if case else re.IGNORECASE)) if query else []
        identity = [selection, record_index, text_field, query, case]
        index = (search or {}).get("index", 0) if (search or {}).get("identity") == identity else 0
        if matches and ctx.triggered_id in ("previous", "next", "query") and (search or {}).get("identity") == identity:
            index = (index + (-1 if ctx.triggered_id == "previous" else 1)) % len(matches)
        index = index % len(matches) if matches else 0
        children = text
        if matches:
            match = matches[index]
            children = [text[: match.start()], html.Mark(text[match.start() : match.end()], id="current-search-match"), text[match.end() :]]
        count = f"{index + 1} of {len(matches)} matches" if matches else ("No matches" if query else "")
        request = {"kind": "end" if ctx.triggered_id == "end" else "match" if matches else "reset", "nonce": [identity, previous, following, submit, end]}
        return details, children, count, {"identity": identity, "index": index}, request

    app.clientside_callback("function(request) { return window.dash_clientside.treeVisualizer.scrollText(request); }", Output("scroll-result", "data"), Input("scroll-request", "data"))
    return app
