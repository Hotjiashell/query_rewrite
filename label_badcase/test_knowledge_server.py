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

    def test_multiuser_settings_and_record_versions_are_independent(self):
        demo = server.load(Path(__file__).parent/'demo_data.json')
        records = server.build_dataset(demo['badcases'], {'records': []}, demo['cases'])
        with tempfile.TemporaryDirectory() as folder:
            store = KnowledgeStore(records, Path(folder)/'annotations.json', {})
            store.mutate({'action':'annotation', 'key':records[0]['key'], 'record_revision':0,
                          'annotation':{'can_solve':True, 'owner':'甲'}})
            settings = {'action':'settings', 'key':records[0]['key'], 'settings_revision':0,
                        'settings':{'prompt':'测试', 'knowledge':'知识'}}
            store.mutate(settings)
            with self.assertRaises(RuntimeError): store.mutate(settings)
            self.assertEqual(store.state['record_settings'][records[0]['key']]['knowledge'], '知识')
            self.assertEqual(store.state['annotations'][records[0]['key']]['owner'], '甲')
            self.assertEqual(store.state['record_settings'][records[0]['key']]['revision'], 1)

if __name__ == '__main__': unittest.main()
