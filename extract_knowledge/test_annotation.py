"""Offline tests for annotation persistence and export."""
import copy
from pathlib import Path
import tempfile
import unittest
from .annotate import Review
from .pipeline import save, read_json


class ReviewTests(unittest.TestCase):
    def fixture(self, root):
        save(root/'knowledge.json', [{'keyword': '会议', 'knowledge': '会议：预订时考虑会议系统',
                                    'reason': '用户需求', 'cluster_id': 'c', 'example_ids': ['r']}])
        save(root/'existing_knowledge.json', {'content': '1. WebG：外网访问\n2. Outlook：邮箱问题\n'})
        save(root/'06_retained_clusters.json', [{'id': 'c', 'records': [{'id': 'r',
             'gt_case_title': '会议预订', 'chat_content': '用户：预订会议', 'queries': ['预订会议']}]}])

    def test_initial_existing_sources_draft_restore_export_and_conflict(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            review = Review(root)
            self.assertEqual([r['text'] for r in review.state['confirmed']], ['WebG：外网访问', 'Outlook：邮箱问题'])
            self.assertEqual(review.candidates[0]['records'][0]['chat_content'], '用户：预订会议')
            state = copy.deepcopy(review.state)
            state['drafts']['0'] = '修改后的会议知识'
            state['statuses']['0'] = 'added'
            state['confirmed'][0]['text'] = '修改后的已有知识'
            state['confirmed'].append({'id': 'new', 'text': '修改后的会议知识', 'source': 'model', 'candidate': '0'})
            result = review.update(state, export=True)
            self.assertEqual(result['revision'], 1)
            self.assertEqual((root/'final_knowledge.txt').read_text(),
                             '1. 修改后的已有知识\n2. Outlook：邮箱问题\n3. 修改后的会议知识\n')
            restored = Review(root)
            self.assertEqual(restored.state['drafts']['0'], '修改后的会议知识')
            self.assertEqual(len(read_json(root/'final_knowledge.json')['knowledge']), 3)
            with self.assertRaisesRegex(ValueError, '另一个页面'):
                review.update(state)
            invalid = copy.deepcopy(restored.state)
            invalid['confirmed'][0]['text'] = ' '
            with self.assertRaisesRegex(ValueError, '不能为空'):
                restored.update(invalid)
            save(root/'knowledge.json', [])
            with self.assertRaisesRegex(ValueError, '已变化'):
                Review(root)

    def test_no_model_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save(root/'knowledge.json', [])
            save(root/'existing_knowledge.json', {'content': ''})
            review = Review(root)
            self.assertEqual(review.candidates, [])
            review.update(review.state, export=True)
            self.assertEqual(read_json(root/'final_knowledge.json')['knowledge'], [])


if __name__ == '__main__':
    unittest.main()
