import json
from pathlib import Path
import tempfile
import argparse
import unittest
from unittest.mock import patch
import cli


class AdapterTests(unittest.TestCase):
    def test_package_provenance_and_parse(self):
        self.assertEqual(cli.package_check()['status'],'PASS')

    def test_relocation_preserves_relative_asset_path(self):
        self.assertEqual(cli.rebase(str(cli.SOURCE_ROOT/'vla-merge-runtime/model.pt'),'/mount'),Path('/mount/vla-merge-runtime/model.pt'))
        with self.assertRaises(ValueError):cli.rebase('/outside/model.pt','/mount')

    def test_missing_assets_fail_closed(self):
        with tempfile.TemporaryDirectory() as root:
            result=cli.runtime_preflight('fastwam','spatial',root)
            self.assertEqual(result['status'],'BLOCKED')
            self.assertGreater(len(result['missing']),0)

    def test_cpu_environment_cannot_use_cuda_or_download(self):
        for backend in ('openvla','fastwam'):
            env=cli.runtime_environment(backend,Path('/mount'))
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'')
            self.assertEqual(env['HF_HUB_OFFLINE'],'1')
            self.assertEqual(env['TRANSFORMERS_OFFLINE'],'1')

    def test_busy_gpu_never_launches_or_hashes_large_weights(self):
        with tempfile.TemporaryDirectory() as root:
            args=argparse.Namespace(host_root=Path(root),backend='openvla',suite='spatial',
                gpu=5,output=Path(root)/'new-output')
            busy={'gpu':5,'uuid':'test','used_mib':1024,'free_mib':80000,'compute_pids':['123']}
            with patch.object(cli,'runtime_preflight',return_value={'status':'PASS'}) as check, \
                 patch.object(cli,'gpu_snapshot',return_value=busy), \
                 patch.object(cli.subprocess,'run') as launch:
                result=cli.smoke(args)
                self.assertEqual(result['status'],'BLOCKED')
                launch.assert_not_called()
                self.assertEqual(check.call_args.kwargs,{'verify_weights':False})


if __name__=='__main__':unittest.main()
