"""Small independent global TIES oracle and safetensors export tests; CPU only."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from pathlib import Path
import tempfile
import unittest
import ties_subset as adapter

np, torch, core, fastlane = adapter.cpu_imports()

def oracle(vectors):
    trimmed = []
    for vector in vectors:
        count = len(vector)
        threshold = sorted(abs(float(x)) for x in vector)[count-int(count*.3)-1]
        row = np.asarray(vector, dtype=np.float64)
        trimmed.append(np.where(np.abs(row) >= threshold, row, 0.))
    stack = np.stack(trimmed)
    signs = np.sign(stack.sum(axis=0))
    fallback = np.sign(signs.sum())
    if fallback:
        signs[signs == 0] = fallback
    picked = np.where(np.where(signs > 0, stack > 0, stack < 0), stack, 0.)
    return picked.sum(axis=0) / np.maximum(np.count_nonzero(picked, axis=0), 1)

class Tests(unittest.TestCase):
    def test_two_and_three_expert_global_oracle(self):
        rng = np.random.default_rng(215)
        for count in [2,3]:
            vectors = [rng.integers(-5, 6, size=257).astype(np.float32) for _ in range(count)]
            # Quantized ties and opposing coordinates exercise threshold and sign fallback.
            vectors[1][:24] = -vectors[0][:24]
            actual, metadata = core.ties_direction(vectors, keep_fraction=.3, chunk_size=31)
            np.testing.assert_array_equal(actual, oracle(vectors))
            self.assertEqual(len(metadata['thresholds']), count)
            self.assertEqual(metadata['selection'], 'exact_ieee_float32_radix_global')

    def test_export_preserves_base_and_original_dtypes(self):
        from safetensors.torch import save_file, load_file
        values = {'a': torch.arange(12, dtype=torch.float32).reshape(3,4),
                  'b': torch.arange(6, dtype=torch.bfloat16).reshape(2,3),
                  'untouched': torch.tensor([11.,22.])}
        direction = np.linspace(-.7,.8,18,dtype=np.float64)
        segments = [{'key':'a','shape':[3,4],'start':0,'end':12},
                    {'key':'b','shape':[2,3],'start':12,'end':18}]
        with tempfile.TemporaryDirectory() as tmp:
            base, out = Path(tmp)/'base.safetensors', Path(tmp)/'out.safetensors'
            save_file(values, str(base)); original_hash = adapter.sha(base)
            adapter.patch_export(base,out,segments,direction,.9,(np,torch,core,fastlane))
            actual = load_file(str(out))
            self.assertEqual(original_hash,adapter.sha(base))
            self.assertTrue(torch.equal(actual['untouched'],values['untouched']))
            for row in segments:
                key = row['key']; delta=torch.from_numpy(direction[row['start']:row['end']]).reshape(row['shape'])
                expected=(values[key].double()+.9*delta).to(values[key].dtype)
                self.assertTrue(torch.equal(actual[key],expected))
                self.assertEqual(actual[key].dtype,values[key].dtype)
        self.assertFalse(torch.cuda.is_initialized())

if __name__ == '__main__': unittest.main()
