"""CPU-only reconstruction, preservation, and viewer smoke checks."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from treehca.visualizer.data import FileCache, discover_files, graph_elements, load_file


def write_rows(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def row(node_id, ancestors=None, prompt="task", **extra):
    return {"node_path": [node_id, *(ancestors or []), "root"], "input": prompt, "output": "<action>search[desk]</action>", **extra}


def test_grouping_preserves_every_field_and_duplicate_record(tmp_path):
    records = [row("a", future={"nested": [None, True, 1.23456789012345]}), row("b"), row("c", ["a"], prompt="next"), row("d", prompt="other"), row("a", score=2)]
    document = load_file(write_rows(tmp_path / "1.jsonl", records))
    assert len(document.trees) == 2
    tree = document.trees[0]
    assert tree.nodes["root"].children == ["a", "b"]
    assert tree.nodes["a"].children == ["c"]
    assert tree.nodes["a"].records == [records[0], records[4]]
    assert tree.nodes["a"].lines == [1, 5]
    assert tree.statistics["saved_records"] == 4
    assert any("multiple" in warning for warning in tree.warnings)


def test_saved_ids_separate_identical_prompts(tmp_path):
    document = load_file(write_rows(tmp_path / "1.jsonl", [row("a", uid="episode-a"), row("b", uid="episode-b")]))
    assert [t.uid for t in document.trees] == ["episode-a", "episode-b"]
    assert not document.trees[0].warnings


def test_missing_ancestor_and_out_of_order_nodes(tmp_path):
    document = load_file(write_rows(tmp_path / "1.jsonl", [row("c", ["b", "a"]), row("a")]))
    tree = document.trees[0]
    assert tree.nodes["b"].records == []
    assert tree.nodes["b"].children == ["c"]
    assert tree.statistics["missing_ancestors"] == 1


@pytest.mark.parametrize("path", [["a", "a", "root"], ["a"], ["root", "a"], [1, "root"]])
def test_invalid_paths_report_line(tmp_path, path):
    with pytest.raises(ValueError, match="line 1"):
        load_file(write_rows(tmp_path / "1.jsonl", [{"node_path": path}]))


def test_conflicting_ancestry_rejected(tmp_path):
    with pytest.raises(ValueError, match="conflicting ancestry"):
        load_file(write_rows(tmp_path / "1.jsonl", [row("a"), row("b", ["a"]), row("b", ["c", "a"])]))


def test_cache_refresh_and_natural_discovery(tmp_path):
    path = write_rows(tmp_path / "2.jsonl", [row("a")])
    write_rows(tmp_path / "10.jsonl", [row("z")])
    cache = FileCache(tmp_path, capacity=1)
    assert [p.name for p in discover_files(tmp_path)] == ["2.jsonl", "10.jsonl"]
    before = cache.load("2.jsonl")
    write_rows(path, [row("a"), row("b")])
    assert cache.load("2.jsonl").record_count == 2
    assert before.record_count == 1
    cache.load("10.jsonl")
    assert list(cache.entries) == ["10.jsonl"]
    with pytest.raises(ValueError, match="Unknown"):
        cache.load("../2.jsonl")


def test_layout_follows_ancestry_and_handles_nonfinite_metrics(tmp_path):
    tree = load_file(write_rows(tmp_path / "1.jsonl", [row("a", score="-Infinity"), row("b", ["a"], score=0.2), row("c", score=0.8)])).trees[0]
    elements, limits = graph_elements(tree, "score")
    nodes = {e["data"]["id"]: e for e in elements if "position" in e}
    assert limits == (0.2, 0.8)
    assert nodes["a"]["position"]["y"] == nodes["c"]["position"]["y"] == 110
    assert nodes["b"]["position"]["y"] == 220
    assert nodes["a"]["data"]["color"] == "#e2e8f0"
    assert nodes["b"]["data"]["color"] != nodes["c"]["data"]["color"]
    assert len(elements) == 7


def test_partial_json_reports_error_and_can_be_retried(tmp_path):
    path = tmp_path / "1.jsonl"
    path.write_text('{"node_path":', encoding="utf-8")
    cache = FileCache(tmp_path)
    with pytest.raises(ValueError, match="line 1"):
        cache.load("1.jsonl")
    write_rows(path, [row("a")])
    assert cache.load("1.jsonl").record_count == 1


def test_page_type_colors_and_legend(tmp_path):
    from treehca.visualizer.app import create_app
    from treehca.visualizer.data import page_type_style

    rows = [row("a", page_type=""), row("b", ["a"], page_type="search_results"), row("c", ["b", "a"], page_type="item_page"), row("d", ["c", "b", "a"], page_type="item_sub_page"), row("e", ["c", "b", "a"]), row("f", ["c", "b", "a"], page_type="future_page")]
    tree = load_file(write_rows(tmp_path / "1.jsonl", rows)).trees[0]
    elements, limits = graph_elements(tree, "page_type")
    colors = {e["data"]["id"]: e["data"]["color"] for e in elements if "position" in e}
    assert limits is None
    assert len({colors[uid] for uid in ("root", "a", "b", "c", "d", "e")}) == 6
    assert page_type_style(tree.nodes["a"])[0] == 'Initial search ("")'
    assert colors["f"] == page_type_style(tree.nodes["f"])[1]
    app = create_app(tmp_path)
    select = next(v["callback"].__wrapped__ for k, v in app.callback_map.items() if "metric.options" in k)
    assert {"label": "Page type", "value": "page_type"} in select("1.jsonl", 0, 0)[-1]
    render = next(v["callback"].__wrapped__ for k, v in app.callback_map.items() if "legend.children" in k)
    _, _, legend = render({"file": "1.jsonl", "tree": 0}, "page_type")
    labels = [entry.children[-1] for entry in legend[:-1]]
    assert {"item_page", "item_sub_page", "search_results", "Missing page type", "Synthetic root", "future_page"} <= set(labels)


def test_dash_layout_and_file_callback(tmp_path):
    pytest.importorskip("dash_cytoscape")
    from treehca.visualizer.app import create_app

    write_rows(tmp_path / "1.jsonl", [row("a", extra="<script>literal</script>")])
    app = create_app(tmp_path)
    client = app.server.test_client()
    for url in ("/", "/_dash-layout", "/_dash-dependencies", "/assets/tree_visualizer.css", "/assets/text_navigation.js"):
        assert client.get(url).status_code == 200
    callback = next(v["callback"].__wrapped__ for k, v in app.callback_map.items() if "file-stats.children" in k)
    options, selected, error, stats = callback("1.jsonl", 0)
    assert len(options) == 1 and selected == 0 and error == ""
    path = tmp_path / "1.jsonl"
    path.write_text("bad json\n", encoding="utf-8")
    assert "line 1" in callback("1.jsonl", 1)[2]


@pytest.mark.parametrize("history", [False, True])
def test_prompt_sections_exclude_chat_wrapper_and_preserve_content(history):
    from agent_system.environments.prompts.webshop import WEBSHOP_TEMPLATE, WEBSHOP_TEMPLATE_NO_HIS
    from treehca.visualizer.presentation import prompt_sections, visible_fields

    history_text = "[Observation 1: ''Search'', Action 1: 'search[desk]']"
    template = WEBSHOP_TEMPLATE if history else WEBSHOP_TEMPLATE_NO_HIS
    body = template.format(task_description="A child's desk under $50", current_observation="'Desk' [SEP] 'Buy Now'", available_actions="'click[buy now]',", step_count=1, history_length=1, action_history=history_text, current_step=2)
    prompt = "system\nWrapper\nuser\n" + body + "\nassistant\n"
    sections, error = prompt_sections(prompt)
    assert error is None
    assert sections["shopping_task"] == "A child's desk under $50"
    assert sections["current_observation"] == "'Desk' [SEP] 'Buy Now'"
    assert sections["available_actions"] == ["click[buy now]"]
    assert (history_text in sections["history"]) if history else sections["history"] is None
    assert visible_fields({"input": prompt, "node_uid": "a", "parent_node_uid": "root", "node_path": ["a", "root"], "children": ["b"], "uid": "g", "score": 0.5}) == {"score": 0.5}


def test_malformed_prompt_is_explicit_not_silently_misparsed():
    from treehca.visualizer.presentation import prompt_sections

    sections, error = prompt_sections("unrecognized input")
    assert error and all(value is None for value in sections.values())


def test_node_callback_hides_ids_and_splits_prompt(tmp_path):
    from agent_system.environments.prompts.webshop import WEBSHOP_TEMPLATE_NO_HIS
    from treehca.visualizer.app import create_app

    prompt = WEBSHOP_TEMPLATE_NO_HIS.format(task_description="desk", current_observation="'Search'", available_actions="'search[<your query>]',")
    write_rows(tmp_path / "1.jsonl", [row("secret-node-uid", prompt=prompt, parent_node_uid="root")])
    app = create_app(tmp_path)
    key, callback = next((k, v) for k, v in app.callback_map.items() if "details.children" in k)
    values = [{"file": "1.jsonl", "tree": 0, "node": "secret-node-uid"}, 0, "output", "", [], 0, 0, 0, 0, []]
    response = app.server.test_client().post(
        "/_dash-update-component",
        json={
            "output": key,
            "outputs": [{"id": o.component_id, "property": o.component_property} for o in callback["output"]],
            "inputs": [{**spec, "value": value} for spec, value in zip(callback["inputs"], values)],
            "state": [{**callback["state"][0], "value": None}],
            "changedPropIds": ["selection.data"],
        },
    )
    assert response.status_code == 200
    payload = response.get_json()["response"]
    details = json.dumps(payload["details"])
    assert "secret-node-uid" not in details and "JSONL lines" not in details
    assert "parent_node_uid" not in details and '"input"' not in details
    context = json.dumps(payload["context"])
    for field in ("shopping_task", "history", "current_observation", "available_actions"):
        assert field in context


def test_field_filters_and_pin_order_across_trees(tmp_path):
    from treehca.visualizer.app import create_app
    from treehca.visualizer.presentation import detail_fields

    records = [
        row("a", score=1, rewards=1, advantages=[1], values=[2], advantage=1, value=2, sampled_expansion_count=4, branch_logit=0.5, webshop_session_id="123", webshop_task_id=7),
        row("b", prompt="other task", score=0),
    ]
    path = write_rows(tmp_path / "1.jsonl", records)
    trees = load_file(path).trees
    pins = ["rewards", "score", "child count"]
    for tree in trees:
        node = next(n for n in tree.nodes.values() if n.records)
        fields = detail_fields(node, node.records[0], pins)
        assert list(fields)[:3] == pins
        assert not {"advantages", "values", "webshop_session_id"} & fields.keys()
    assert fields["rewards"] == "Not saved for this node"
    assert fields["score"] == 0
    unpinned = detail_fields(trees[0].nodes["a"], records[0], ["score"])
    assert next(iter(unpinned)) == "score"
    assert unpinned["advantage"] == 1 and unpinned["value"] == 2

    app = create_app(tmp_path)
    select = next(v["callback"].__wrapped__ for k, v in app.callback_map.items() if "metric.options" in k)
    options = select("1.jsonl", 0, 0)[-1]
    metrics = {o["value"] for o in options}
    assert not {"sampled_expansion_count", "branch_logit", "webshop_session_id", "webshop_task_id"} & metrics
    assert {"score", "rewards", "advantage", "value"} <= metrics
    # Pins live in a store updated only by the double-click handler.
    assert not any("pinned-fields" in key for key in app.callback_map)
    layout = app.server.test_client().get("/_dash-layout").get_json()

    def find(component, target):
        if isinstance(component, dict):
            if component.get("props", {}).get("id") == target:
                return component
            for value in component.values():
                if result := find(value, target):
                    return result
        elif isinstance(component, list):
            for value in component:
                if result := find(value, target):
                    return result
        return None

    assert find(layout, "pinned-fields")["type"] == "Store"
    assert find(layout, "record")["type"] == "Store"


def test_double_click_toggles_field_pins():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is needed to exercise the field pin handler")
    asset = Path(__file__).resolve().parents[2] / "treehca/visualizer/assets/field_pins.js"
    script = """
const assert = require('node:assert/strict');
const listeners = {}, updates = [];
global.document = {addEventListener: (name, callback) => listeners[name] = callback};
global.window = {dash_clientside: {set_props: (id, props) => {assert.equal(id, 'pinned-fields'); updates.push(props.data);}}};
require(process.argv[1]);
function fire(name, field, key, repeat=false) {
  listeners[name]({type:name, key, repeat, preventDefault:()=>{}, target:{closest:()=>field===null ? null : {getAttribute:()=>field}}});
}
assert.equal(listeners.click, undefined);
fire('dblclick','score');
fire('dblclick','rewards');
assert.deepEqual(updates.at(-1), ['score','rewards']);
// A new DOM node with the same field name can unpin after tree navigation.
fire('dblclick','score');
assert.deepEqual(updates.at(-1), ['rewards']);
fire('dblclick',null);
fire('keydown','score','x');
fire('keydown','score','Enter',true);
assert.equal(updates.length, 3);
fire('keydown','score','Enter');
assert.deepEqual(updates.at(-1), ['rewards','score']);
fire('keydown','score',' ');
assert.deepEqual(updates.at(-1), ['rewards']);
"""
    subprocess.run([node, "-e", script, str(asset)], check=True, capture_output=True, text=True)


def test_subtree_viewport_fits_descendants_and_single_leaf():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is needed to exercise the clientside viewport helper")
    asset = Path(__file__).resolve().parents[2] / "treehca/visualizer/assets/viewport.js"
    script = """
const assert = require('node:assert/strict');
global.window = {dash_clientside: {callback_context: {triggered_id: 'fit-subtree'}, no_update: 'unchanged'}};
global.document = {getElementById: () => ({clientWidth: 800, clientHeight: 400})};
require(process.argv[1]);
const elements = [
  {data: {id:'root'}, position:{x:500,y:0}},
  {data: {id:'a'}, position:{x:0,y:110}},
  {data: {id:'b'}, position:{x:0,y:220}},
  {data: {id:'sibling'}, position:{x:1000,y:110}},
  {data: {source:'root',target:'a'}},
  {data: {source:'a',target:'b'}},
  {data: {source:'root',target:'sibling'}}
];
const fit = window.dash_clientside.rolloutViewport.fit;
const [zoom, pan] = fit(0,1,elements,{node:'a'});
assert.ok(zoom > 1);
assert.equal(pan.x, 400);
assert.ok(Math.abs(pan.y + zoom * 165 - 200) < 1e-8);
for (const y of [110,220]) {
  assert.ok((y-29)*zoom+pan.y >= 35.999);
  assert.ok((y+29)*zoom+pan.y <= 364.001);
}
const [leafZoom, leafPan] = fit(0,2,elements,{node:'b'});
assert.ok(leafZoom > zoom);
assert.ok(Math.abs(leafPan.y + leafZoom * 220 - 200) < 1e-8);
window.dash_clientside.callback_context.triggered_id = 'fit';
const [wholeZoom] = fit(1,2,elements,{node:'a'});
assert.ok(wholeZoom < zoom);
window.dash_clientside.callback_context.triggered_id = 'fit-subtree';
assert.deepEqual(fit(0,0,elements,null), ['unchanged','unchanged']);
"""
    subprocess.run([node, "-e", script, str(asset)], check=True, capture_output=True, text=True)
