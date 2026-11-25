"""Regression checks for both rollout formats and local HTTP routes."""
import json
import tempfile
import threading
import unittest
from pathlib import Path
from http.server import ThreadingHTTPServer
from urllib.request import urlopen
from urllib.error import HTTPError

from .data import read_document
from .__main__ import make_handler


def row(output='Peru', **extra):
    return {'input': 'Your question: Where is Lima?', 'output': f'<answer>{output}</answer>', 'score': 0, **extra}


def search_row(step, trajectory='t', **extra):
    return {'input': f'Your question: Where is Lima?\n\nPrior steps: {step}',
            'output': '<think>Look it up.</think><search>Lima</search>', 'id': f'{trajectory}-{step}',
            'previous_step_ids': [f'{trajectory}-{i}' for i in range(step)],
            'trajectory_id': trajectory, **extra}


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / '2.jsonl'

    def read(self, rows):
        self.path.write_text('\n'.join(json.dumps(r) for r in rows))
        return read_document(self.path)

    def test_saved_ids_keep_duplicate_samples_and_out_of_order_parents(self):
        doc = self.read([{**r, 'group_id': 'training-group'} for r in [row('Peru', id='c', previous_step_ids=['a'], turn_reward=-2.5), row(id='a', previous_step_ids=[]), row(id='b', previous_step_ids=[])]])
        nodes = doc['groups'][0]['nodes']
        self.assertEqual(len(nodes), 4)
        self.assertEqual(nodes[1]['parent'], nodes[2]['id'])
        self.assertEqual(nodes[1]['records'][0]['turn_reward'], -2.5)
        self.assertNotEqual(nodes[2]['id'], nodes[3]['id'])

    def test_same_problem_different_training_groups_remain_separate(self):
        rows = [row(id=f'{g}-{t}', previous_step_ids=[], trajectory_id=f'{g}-{t}', group_id=g)
                for g in ('sample-1', 'sample-2') for t in range(2)]
        doc = self.read(rows)
        self.assertEqual([g['record_count'] for g in doc['groups']], [2, 2])
        self.assertTrue(all(g['grouping'] == 'Training group ID' for g in doc['groups']))

    def test_trajectory_ids_prevent_prompt_merging(self):
        for make in (lambda t: row(id=t, previous_step_ids=[], trajectory_id=t),
                     lambda t: search_row(0, t)):
            with self.subTest(make=make):
                doc = self.read([make('a'), make('b')])
                self.assertEqual(len(doc['groups']), 2)
                self.assertTrue(all(g['grouping'] == 'Trajectory ID' for g in doc['groups']))

    def test_search_shuffled_full_trajectory_survives_limited_prompt_history(self):
        rows = [search_row(i) for i in range(50)]
        rows[20]['output'] = 'Invalid output without an action tag'
        rows[-1]['turn_reward'] = 1
        rows[-1]['gigpo_turn_advantage'] = 0.25
        doc = self.read(list(reversed(rows)))
        self.assertEqual(len(doc['groups']), 1)
        group = doc['groups'][0]
        self.assertEqual(group['context'], rows[0]['input'])
        nodes = {n['records'][0]['id']: n for n in group['nodes'] if n['records']}
        self.assertEqual(len(nodes), 50)
        self.assertEqual(nodes['t-0']['parent'], 'root')
        for i in range(1, 50):
            self.assertEqual(nodes[f't-{i}']['parent'], nodes[f't-{i-1}']['id'])
        self.assertEqual(nodes['t-20']['label'], '[No action tag]')
        self.assertEqual(nodes['t-49']['records'][0]['gigpo_turn_advantage'], 0.25)
        self.assertFalse(any('absent' in w for w in group['warnings']))

    def test_parent_ids_work_without_trajectory_ids_or_matching_prompts(self):
        rows = [search_row(i) for i in (2, 0, 1)]
        for r in rows:
            del r['trajectory_id']
        doc = self.read(rows + [row(id='other-root', previous_step_ids=[])])
        self.assertEqual([g['record_count'] for g in doc['groups']], [3, 1])
        self.assertEqual(doc['groups'][0]['grouping'], 'Saved ancestry')

    def test_missing_ancestors_are_reported_and_do_not_invent_steps(self):
        doc = self.read([search_row(3), search_row(2)])
        group = doc['groups'][0]
        self.assertEqual(group['record_count'], 2)
        self.assertEqual(len(group['nodes']), 3)
        self.assertEqual(group['context'], search_row(2)['input'])
        self.assertTrue(any('parent t-1 is absent from the file' in w for w in group['warnings']))
        self.assertTrue(any('earlier ancestor' in w for w in group['warnings']))

    def test_partial_metadata_does_not_downgrade_complete_links(self):
        doc = self.read([search_row(1, group_id='g'), search_row(0),
                         {'input': 'unrelated prompt', 'output': '<action>look</action>', 'group_id': 'g'}])
        group = doc['groups'][0]
        self.assertEqual(len(doc['groups']), 1)
        self.assertEqual(group['record_count'], 3)
        self.assertEqual(group['nodes'][1]['parent'], group['nodes'][2]['id'])
        self.assertTrue(any('incomplete ancestry' in w for w in group['warnings']))

    def test_conflicting_group_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Conflicting group IDs'):
            self.read([search_row(0, group_id='a'), search_row(1, group_id='b')])

    def test_duplicate_ids_across_prompts_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate record IDs'):
            self.read([search_row(0), {**search_row(1), 'id': 't-0'}])

    def test_inconsistent_ancestry_is_reported(self):
        doc = self.read([search_row(0), search_row(1, previous_step_ids=[]), search_row(2)])
        self.assertTrue(any('inconsistent saved ancestry' in w for w in doc['groups'][0]['warnings']))

    def test_legacy_search_does_not_invent_or_merge_turns(self):
        record = {'input': search_row(0)['input'], 'output': '<action>look</action>'}
        group = self.read([record, record])['groups'][0]
        self.assertEqual(group['grouping'], 'Prompt fallback')
        self.assertEqual(len(group['nodes']), 3)
        self.assertTrue(all(n['parent'] == 'root' for n in group['nodes'][1:]))

    def test_null_ids_use_legacy_fallback(self):
        group = self.read([row(id=None, trajectory_id=None, group_id=None), row()])['groups'][0]
        self.assertEqual(group['grouping'], 'Prompt fallback')
        self.assertEqual(len(group['nodes']), 3)

    def test_trajectory_without_turn_ids_keeps_independent_records(self):
        group = self.read([row(trajectory_id='t'), row(trajectory_id='t')])['groups'][0]
        self.assertEqual(group['grouping'], 'Trajectory ID')
        self.assertEqual(group['mode'], 'Independent saved turns')
        self.assertEqual(len(group['nodes']), 3)

    def test_distinct_problems_are_separate_groups(self):
        self.assertEqual(len(self.read([row(), {**row(), 'input': 'A second question'}])['groups']), 2)

    def test_cycle_is_reported(self):
        with self.assertRaisesRegex(ValueError, 'Cycle'):
            self.read([row(id='a', previous_step_ids=['b']), row(id='b', previous_step_ids=['a'])])

    def test_bad_and_partial_lines_preserve_valid_records(self):
        self.path.write_text(json.dumps(row()) + '\nnot json\n{"input":')
        doc = read_document(self.path)
        self.assertEqual(len(doc['warnings']), 2)
        self.assertEqual(doc['groups'][0]['record_count'], 1)

    def test_http_routes_and_directory_boundary(self):
        self.read([row()])
        (self.path.parent / '10.jsonl').write_text('')
        server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(self.path.parent))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f'http://127.0.0.1:{server.server_port}'
        try:
            with urlopen(base + '/api/files') as response:
                self.assertEqual(json.load(response)['files'], ['2.jsonl', '10.jsonl'])
            with urlopen(base + '/api/file?name=2.jsonl') as response:
                self.assertEqual(json.load(response)['groups'][0]['record_count'], 1)
            with urlopen(base + '/') as response:
                self.assertIn(b'Training Rollout Trees', response.read())
            with urlopen(base + '/decision_model.js') as response:
                self.assertEqual(response.headers.get_content_type(), 'text/javascript')
                self.assertIn(b'DecisionModelView', response.read())
            with self.assertRaises(HTTPError) as error:
                urlopen(base + '/api/file?name=../outside.jsonl')
            self.assertEqual(error.exception.code, 400)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == '__main__':
    unittest.main()
