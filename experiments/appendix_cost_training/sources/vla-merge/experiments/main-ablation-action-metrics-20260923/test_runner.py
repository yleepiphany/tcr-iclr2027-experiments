import unittest
from unittest.mock import patch
import tempfile
from pathlib import Path
import run_metrics as r


class Contract(unittest.TestCase):
    def test_exact_six(self):
        models = r.models()
        self.assertEqual(len(models), 6)
        self.assertEqual({v['group'] for v in models.values()}, {'last_call', 'expert_prefix'})

    def test_missing_model_rejected(self):
        data = r.read(r.FORMAL)
        data['jobs'] = [j for j in data['jobs'] if j['model_name'] != 'last_call-r03']
        with patch.object(r, 'read', return_value=data):
            with self.assertRaises(ValueError):
                r.models()

    def test_inconsistent_identity_rejected(self):
        data = r.read(r.FORMAL)
        for j in data['jobs']:
            if j['model_name'] == 'last_call-r01':
                j['model_sha256'] = 'wrong'
                break
        with patch.object(r, 'read', return_value=data):
            with self.assertRaises(ValueError):
                r.models()

    def test_no_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / 'started.json').touch()
            with patch.object(r, 'RUN', path), patch.object(r, 'frozen', return_value={}):
                with self.assertRaises(FileExistsError):
                    r.dispatch()

    def test_card_lock_excludes_second_owner(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lock = r.core.card_flock.take_card(2, 'test-uuid', 'test', 'cpu', root)
            try:
                self.assertIsNone(r.core.card_flock.take_card(2, 'test-uuid', 'other', 'cpu', root))
            finally:
                lock.release()
            self.assertTrue(r.core.card_flock.card_path(2, 'test-uuid', root).exists())
            again = r.core.card_flock.take_card(2, 'test-uuid', 'other', 'cpu', root)
            self.assertIsNotNone(again)
            again.release()


if __name__ == '__main__':
    unittest.main()
