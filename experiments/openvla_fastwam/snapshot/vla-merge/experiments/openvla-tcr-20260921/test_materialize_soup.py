import json
from pathlib import Path
import tempfile
import unittest
import torch
from safetensors.torch import save_file,load_file
from materialize_soup import mean_tensors,merge_stats,build,sha,ORDER,SHARED

class SoupTests(unittest.TestCase):
    def test_dense_average(self):
        x=[torch.full((2,3),i,dtype=torch.bfloat16) for i in range(4)]
        y=mean_tensors(x)
        self.assertEqual(y.dtype,torch.bfloat16)
        self.assertTrue(torch.equal(y,torch.full_like(y,1.5)))
        self.assertTrue(torch.equal(x[0],torch.zeros_like(x[0])))
    def test_rejections(self):
        with self.assertRaises(ValueError):mean_tensors([torch.ones(2)]*3)
        with self.assertRaises(ValueError):mean_tensors([torch.ones(2)]*3+[torch.ones(3)])
        with self.assertRaises(ValueError):mean_tensors([torch.ones(2)]*3+[torch.full((2,),float('nan'))])
        with self.assertRaises(ValueError):mean_tensors([torch.tensor([i]) for i in range(4)])
    def test_integer_and_stats(self):
        self.assertTrue(torch.equal(mean_tensors([torch.tensor([4])]*4),torch.tensor([4])))
        self.assertEqual(merge_stats([{'a':[1]},{'b':[2]}]),{'a':[1],'b':[2]})
        with self.assertRaises(ValueError):merge_stats([{'a':[1]},{'a':[2]}])
    def test_complete_export(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);experts={};files={}
            for i,name in enumerate(ORDER):
                p=root/name;p.mkdir();experts[name]={'local_path':str(p)}
                for n in SHARED:(p/n).write_text('{\n}' if n=='config.json' and i==1 else '{}')
                (p/'dataset_statistics.json').write_text(json.dumps({name:{'mean':[i]}}))
                index={'metadata':{'total_size':24},'weight_map':{'weight':'part.safetensors','bias':'part.safetensors'}}
                (p/'model.safetensors.index.json').write_text(json.dumps(index))
                save_file({'weight':torch.full((2,2),float(i)),'bias':torch.full((2,),float(i))},str(p/'part.safetensors'))
                for c in ('action_head','proprio_projector'):torch.save({'weight':torch.full((2,2),float(i))},p/(c+'--1_checkpoint.pt'))
                for f in p.iterdir():files[str(f)]={'sha256':sha(f),'size':f.stat().st_size,'mtime_ns':f.stat().st_mtime_ns}
            ledger=root/'ledger.json';ledger.write_text(json.dumps({'experts':experts,'files':files}))
            result=build(ledger,root/'run',expected_keys=2)
            self.assertFalse(result['is_tcr']);self.assertTrue(result['complete'])
            self.assertTrue(torch.equal(load_file(str(root/'run/checkpoint/part.safetensors'))['weight'],torch.full((2,2),1.5)))
            self.assertEqual(len(json.loads((root/'run/checkpoint/dataset_statistics.json').read_text())),4)
            with self.assertRaises(FileExistsError):build(ledger,root/'run',expected_keys=2)

if __name__=='__main__':unittest.main()
