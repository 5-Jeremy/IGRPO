"""Read saved records and construct trees without inventing trajectory IDs."""
import json
import re
from collections import OrderedDict

def action(record):
    trace = record.get("reasoning_trace")
    if isinstance(trace, dict) and isinstance(trace.get("action"), str):
        return trace["action"]
    # Ignore mentions of operations in the model's reasoning.
    output = re.split(r"</think>", record["output"], flags=re.I)[-1]
    matches = re.finditer(r"<(search|answer|action)>(.*?)</\1>", output, re.S | re.I)
    labels = [m[2].strip() if m[1].lower() == "action" else m[0].strip() for m in matches]
    return " | ".join(label or "[Empty action]" for label in labels) or "[No action tag]"


def identity(value):
    return str(value) if isinstance(value, (str, int)) and not isinstance(value, bool) and str(value) else None


def saved_ancestry(record):
    return (isinstance(record.get('id'), str) and bool(record['id']) and
            isinstance(record.get('previous_step_ids'), list) and
            all(isinstance(p, str) and p for p in record['previous_step_ids']))


def group_rows(rows):
    """Resolve identity across the whole file before considering prompt text.

    Missing ancestor IDs still connect surviving records from the same episode.
    Namespaces keep group, trajectory and turn IDs distinct. Conflicting explicit
    training groups are rejected rather than silently merged via ancestry.
    """
    parents = {}

    def find(key):
        parents.setdefault(key, key)
        root = key
        while parents[root] != root:
            root = parents[root]
        while parents[key] != key:
            next_key = parents[key]
            parents[key] = root
            key = next_key
        return root

    def union(a, b):
        parents[find(b)] = find(a)

    seen_ids = set()
    for line, record in rows:
        anchor = ('line', line)
        find(anchor)
        tokens = []
        for field in ('group_id', 'trajectory_id', 'id'):
            value = identity(record.get(field))
            if value is not None:
                if field == 'id':
                    if value in seen_ids:
                        raise ValueError(f'Duplicate record IDs: {value}')
                    seen_ids.add(value)
                tokens.append((field, value))
        if saved_ancestry(record):
            tokens.extend(('id', p) for p in record['previous_step_ids'])
        if not tokens:
            tokens.append(('prompt', record['input']))
        for token in tokens:
            union(anchor, token)

    groups = OrderedDict()
    for row in rows:
        groups.setdefault(find(('line', row[0])), []).append(row)
    for members in groups.values():
        explicit = {identity(r.get('group_id')) for _, r in members} - {None}
        if len(explicit) > 1:
            raise ValueError('Conflicting group IDs linked by saved trajectory or parent IDs')
    return groups.values()


def read_document(path):
    rows = []
    warnings = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            if not isinstance(record, dict) or not all(isinstance(record.get(k), str) for k in ('input', 'output')):
                raise ValueError('expected an object with string input and output')
        except (ValueError, TypeError) as exc:
            warnings.append(f'Line {line_number}: {exc}')
            continue
        rows.append((line_number, record))
    result = []
    for index, rows in enumerate(group_rows(rows)):
        notes = []
        records = [r for _, r in rows]
        grouping = ('Training group ID' if any(identity(r.get('group_id')) is not None for r in records) else
                    'Trajectory ID' if any(identity(r.get('trajectory_id')) is not None for r in records) else
                    'Saved ancestry' if any(identity(r.get('id')) is not None for r in records) else
                    'Prompt fallback')
        initial = [r for r in records if saved_ancestry(r) and not r['previous_step_ids']]
        representative = initial[0] if initial else min(
            records, key=lambda r: len(r['previous_step_ids']) if saved_ancestry(r) else float('inf'))
        context = representative['input']
        if not initial and grouping != 'Prompt fallback':
            notes.append('Initial turn is not identifiable in the saved records; context shows the earliest identifiable saved prompt.')
        if initial and len({r['input'] for r in initial}) > 1:
            notes.append('Initial prompts differ within this group; context shows one initial prompt. Inspect individual inputs for each rollout.')
        if grouping in ('Trajectory ID', 'Saved ancestry'):
            notes.append('Training group ID unavailable; original training groups cannot be recovered. Grouping uses saved episode identity or ancestry.')
        nodes = [{'id': 'root', 'parent': None, 'label': 'Initial context', 'records': [], 'kind': 'root'}]
        # Partial metadata must not downgrade an entire group to inferred history.
        use_ids = any(identity(r.get('id')) is not None or identity(r.get('trajectory_id')) is not None for r in records)
        if use_ids:
            # Full-path IGRPO saves a distinct record per path through a shared node.
            def graph_id(line, record):
                return f"node-{record['node_uid']}" if record.get("node_uid") else f"n{line}"

            saved_nodes = {}
            ids = {r['id']: graph_id(line, r) for line, r in rows if isinstance(r.get('id'), str)}
            for line, record in rows:
                if not saved_ancestry(record):
                    notes.append(f'Line {line}: incomplete ancestry metadata; shown as an independent saved turn.')
                parents = record['previous_step_ids'] if saved_ancestry(record) else []
                parent = ids.get(str(parents[-1])) if parents else 'root'
                if parent is None:
                    notes.append(f'Line {line}: parent {parents[-1]} is absent from the file; attached to context.')
                missing = [p for p in parents[:-1] if p not in ids]
                if missing:
                    notes.append(f'Line {line}: {len(missing)} earlier ancestor record(s) absent from the file.')
                node_id = graph_id(line, record)
                if node_id not in saved_nodes:
                    node = {'id': node_id, 'parent': parent or 'root', 'label': action(record),
                            'records': [], 'kind': 'saved'}
                    nodes.append(node)
                    saved_nodes[node_id] = node
                node = saved_nodes[node_id]
                if node['parent'] != (parent or 'root'):
                    raise ValueError(f'Conflicting parents for node {node_id}')
                node['records'].append({'line': line, **record})
            by_id = {r['id']: r for r in records if isinstance(r.get('id'), str)}
            for line, record in rows:
                if saved_ancestry(record):
                    for i, previous in enumerate(record['previous_step_ids']):
                        prior = by_id.get(previous)
                        if prior is not None and prior.get('previous_step_ids') != record['previous_step_ids'][:i]:
                            notes.append(f'Line {line}: inconsistent saved ancestry; edges use the immediate saved parent.')
                            break
            # Reject cyclic parent links rather than hanging the browser layout.
            parent_map = {n['id']: n['parent'] for n in nodes}
            for node in nodes:
                seen, current = set(), node['id']
                while current is not None:
                    if current in seen:
                        raise ValueError('Cycle in saved parent IDs')
                    seen.add(current)
                    current = parent_map[current]
        else:
            notes.append("Ancestry is unavailable. Each saved turn is shown independently.")
            for line, record in rows:
                nodes.append({'id': f'n{line}', 'parent': 'root', 'kind': 'saved', 'label': action(record),
                              'records': [{'line': line, **record}]})
        mode = 'Saved parent IDs' if any(saved_ancestry(r) for r in records) else 'Independent saved turns'
        result.append({'context': context, 'grouping': grouping, 'index': index, 'nodes': nodes,
                       'record_count': len(rows), 'mode': mode, 'warnings': notes})
    return {'groups': result, 'warnings': warnings, 'grouping': 'Grouping uses training group IDs, trajectory IDs, saved ancestry, or matching prompts.'}
