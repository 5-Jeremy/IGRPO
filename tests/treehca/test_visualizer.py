"""CPU-only reconstruction, preservation, and viewer smoke checks."""

import json

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
