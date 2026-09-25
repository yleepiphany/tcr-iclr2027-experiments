"""CPU-only artifact and numerical gate before conservative RegMean++ rollout."""
import hashlib
import json
import math
from pathlib import Path
import torch
from safetensors import safe_open

ROOT = Path('/mnt/workspace/Wilson/parameter-fusion/vla-merge-runtime/experiments/static-observation-baselines-20260922/regmeanpp_spectral_smoothing/attempt-01')

def main():
    torch.set_num_threads(2)
    ended = json.loads((ROOT / 'BUILD-ENDED.json').read_text())
    if ended['exit_code'] != 0:
        raise ValueError('Build has not completed successfully')
    checkpoint = ROOT / 'checkpoint'
    manifest = json.loads((checkpoint / 'block_regmeanpp_manifest.json').read_text())
    expected = {'method': 'pi05_regmeanpp_alpha03_static_t1_mean_centered_spectral_smoothing_v1',
                'offdiag_scale': .3, 'ridge_ratio': 0., 'max_correction_ratio': 0.,
                'module_count': 418, 'modified_tensor_count': 422, 'realized_row_total': 2368400,
                'training': False, 'gradient_or_backward': False,
                'replay_prefix': 'merged', 'unreported_extensions_applied': False,
                'strict_original_equivalence': False, 'spectral_regularization_applied': True,
                'demonstration_action_labels_used': False, 'native_denoising_trajectory_used': False,
                'environment_rollout_calibration_used': False}
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f'Completed checkpoint differs: {key}={manifest.get(key)!r}')
    with (checkpoint / 'model.safetensors').open('rb') as stream:
        model_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    if model_sha != manifest['model_sha256']:
        raise ValueError('Checkpoint hash differs')
    residuals = []
    inactive = {}
    for name, metric in manifest['modules'].items():
        if metric.get('offdiag_scale') != .3 or metric.get('solve_dtype') != 'float64/complex128':
            raise ValueError(f'Wrong kernel: {name}')
        if any(metric.get(key) is not False for key in ('soup_centered_ridge', 'pseudo_inverse', 'correction_cap', 'relative_weighting', 'rejection_gate')):
            raise ValueError(f'Forbidden extension: {name}')
        residual = metric['smooth_normal_equation_relative_residual']
        if 'relative_residual' in metric or metric.get('strict_original_equivalence') is not False:
            raise ValueError(f'Ambiguous residual semantics: {name}')
        if not math.isfinite(metric['original_centered_equation_relative_residual']):
            raise ValueError(f'Missing original-equation deviation: {name}')
        if not math.isfinite(residual) or residual > 1e-7:
            raise ValueError(f'Unacceptable residual: {name}')
        residuals.append(residual)
        if metric['exact_zero_energy_coordinates']:
            inactive[name] = metric['exact_zero_energy_coordinates']
    tensor_count = 0
    with safe_open(checkpoint / 'model.safetensors', framework='pt', device='cpu') as tensors:
        for name in tensors.keys():
            tensor = tensors.get_tensor(name)
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(f'Nonfinite stored tensor: {name}')
            tensor_count += 1
    result = {'status': 'passed_cpu_artifact_and_finite_validation', 'model_sha256': model_sha,
              'tensor_count': tensor_count, 'all_stored_tensors_finite': True,
              'module_count': len(residuals), 'max_smoothed_equation_relative_residual': max(residuals),
              'original_equation_deviation_separately_reported': True,
              'zero_support_module_count': len(inactive), 'zero_support_columns': inactive,
              'calibration_observations': 200, 'formal_evaluation_authorized_only_after_normal_nonzero_40episode_gate': True,
              'next_stage': '40-episode gate only; native reload and inference exercised by gate',
              'gpu_initialized_by_this_check': torch.cuda.is_initialized()}
    if result['gpu_initialized_by_this_check']:
        raise AssertionError('CPU check initialized CUDA')
    (ROOT / 'CHECKPOINT-CPU-VALIDATION.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'zero_support_columns'}, indent=2))

if __name__ == '__main__':
    main()
