import unittest
import torch
from static_bank import initial_noise,validate_model_batch,DATA_COLUMNS

def sample():
    d={'tokens':torch.zeros(1,200,dtype=torch.int64),'masks':torch.ones(1,200,dtype=torch.bool),
       'x_t':initial_noise(123),'time':torch.ones(1)}
    for i in range(3):d[f'image_{i}']=torch.zeros(1,3,224,224);d[f'image_mask_{i}']=torch.ones(1,dtype=torch.bool)
    return d

class Tests(unittest.TestCase):
    def test_no_target_columns_or_hidden_labels(self):
        self.assertTrue(all('action' not in k and 'reward' not in k and 'success' not in k for k in DATA_COLUMNS))
        d=sample();validate_model_batch(d);d['action']=torch.zeros(1,14)
        with self.assertRaises(ValueError):validate_model_batch(d)
    def test_three_real_cameras_and_initial_time_only(self):
        d=sample();d['image_mask_2'][0]=False
        with self.assertRaises(ValueError):validate_model_batch(d)
        d=sample();d['time'][0]=.1
        with self.assertRaises(ValueError):validate_model_batch(d)
    def test_reproducible_independent_full_latents(self):
        a=initial_noise(123);self.assertEqual(tuple(a.shape),(1,50,32));self.assertTrue(torch.equal(a,initial_noise(123)))
        self.assertFalse(torch.equal(a,initial_noise(124)))
    def test_nonfinite_inputs_fail(self):
        d=sample();d['x_t'][0,0,0]=float('nan')
        with self.assertRaises(ValueError):validate_model_batch(d)

if __name__=='__main__':unittest.main()
