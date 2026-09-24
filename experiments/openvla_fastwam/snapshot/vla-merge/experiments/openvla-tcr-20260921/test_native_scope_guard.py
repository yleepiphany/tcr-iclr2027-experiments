import unittest
from native_scope_guard import UNUSED_POOL_LINEARS, validate_native_scope


class ScopeTests(unittest.TestCase):
    def test_only_exact_five_pool_modules_allowed(self):
        used = {'action_head.fc', 'backbone.projector.fc1'}
        self.assertEqual(set(validate_native_scope(used | UNUSED_POOL_LINEARS, used)), UNUSED_POOL_LINEARS)
        for missing in used:
            with self.assertRaises(ValueError):
                validate_native_scope(used | UNUSED_POOL_LINEARS, used - {missing})
        with self.assertRaises(ValueError):
            validate_native_scope(used | UNUSED_POOL_LINEARS | {'unknown'}, used)
        with self.assertRaises(ValueError):
            validate_native_scope(used | UNUSED_POOL_LINEARS, used | {'unknown'})


if __name__ == '__main__':
    unittest.main()
