import tempfile
import unittest
from pathlib import Path
from knowledge_server import KnowledgeStore
import server

class KnowledgeTests(unittest.TestCase):
    def test_required_decision_optional_details_and_reload(self):
        demo = server.load(Path(__file__).parent/'demo_data.json')
        records = server.build_dataset(demo['badcases'], {'records': []}, demo['cases'])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'annotations.json'
            store = KnowledgeStore(records, path, {})
            key = records[0]['key']
            body = {'action':'annotation', 'key':key, 'record_revision':0, 'annotation':{}}
            with self.assertRaises(ValueError): store.mutate(body)
            self.assertFalse(path.exists())
            for decision in (True, False):
                body['annotation'] = {'can_solve':decision}
                store.mutate(body)
                body['record_revision'] += 1
            loaded = KnowledgeStore(records, path, {})
            self.assertIs(loaded.state['annotations'][key]['can_solve'], False)
            self.assertEqual(loaded.summary()['completed'], 1)
            with self.assertRaises(RuntimeError):
                loaded.mutate({**body, 'record_revision':0})

if __name__ == '__main__': unittest.main()
