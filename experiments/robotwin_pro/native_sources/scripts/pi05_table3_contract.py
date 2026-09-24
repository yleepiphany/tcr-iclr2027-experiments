"""One-repeat Table-3 controls; standard-library validation and row allocation."""
from collections import Counter
import math

VARIANTS = ('full', 'demo', 'early', 'last', 'expert_prefix', 'uniform', 'action_only', 'conditioning_action')
FLOWS = [0, 5, 9]


def request_selection(metadata, mode):
    requests = sorted({s['request_index'] for s in metadata})
    if len(requests) < 5:
        raise ValueError('Fewer than five distinct requests: preserve failure, do not select by success')
    selected = requests[:5] if mode == 'early' else [requests[(len(requests)-1)*k//4] for k in range(5)]
    slots = []
    for slot, request in enumerate(selected):
        indices = sorted([i for i, s in enumerate(metadata) if s['request_index'] == request],
                         key=lambda i: metadata[i]['flow_index'])
        if [metadata[i]['flow_index'] for i in indices] != FLOWS:
            raise ValueError('Incomplete request/flow triple')
        slots.extend(indices)
    return slots, selected


def row_indices(nrows, cap, flow, request_slot, *, cameras=1, camera=0, last_only=False):
    """One quota per request/module, split across flows AND views without duplicates.

    For time projections nrows=1; alternate which flow supplies that one row.
    The last-only arm places the identical token/view quota at flow 9.
    """
    if nrows < 1 or cap < 1 or flow not in FLOWS or not 0 <= camera < cameras:
        raise ValueError('Invalid row-selection arguments')
    quota = min(cap, nrows*cameras)
    tokens = [0] if quota == 1 else [(nrows*cameras-1)*i//(quota-1) for i in range(quota)]
    result = []
    for i, token in enumerate(tokens):
        stage = 9 if last_only else FLOWS[(i+request_slot) % 3]
        if stage == flow and token//nrows == camera:
            result.append(token % nrows)
    return result


def validate_config(config):
    if config.get('schema') != 'table3_one_repeat_v1' or config.get('variant') not in VARIANTS:
        raise ValueError('Unknown ablation configuration')
    if config.get('row_cap_per_request_module') != 16 or config.get('repeat') != 1:
        raise ValueError('Frozen single-repeat budget differs')
    if config['variant'] != 'full' and not config.get('full_manifest'):
        raise ValueError('Ablations require the same completed Full manifest')


def validate_trace(manifest, dense_path, name):
    from pathlib import Path
    if manifest.get('task') != name or Path(manifest['calibration_policy']).resolve() != Path(dense_path).resolve():
        raise ValueError('Wrong dense expert trace')
    samples = manifest.get('samples', [])
    if len(samples) != 150 or manifest.get('sample_count') != 150:
        raise ValueError('Expected 10 tasks x 5 requests x 3 flows')
    grouped = {}
    for sample in samples:
        grouped.setdefault((sample['prompt_signature'], sample['selected_request_slot']), []).append(sample['flow_index'])
        if sample.get('vision_count') != 3:
            raise ValueError('Missing full three-camera prefix')
    if len(grouped) != 50 or any(sorted(v) != FLOWS for v in grouped.values()):
        raise ValueError('Request/flow groups differ')
    if set(Counter(k[0] for k in grouped).values()) != {5}:
        raise ValueError('Task allocation differs')
    if manifest.get('table3_capture_version') != 1:
        raise ValueError('Legacy cache is not a Table-3 matched input')


def validate_rows(metrics, names, variant):
    solved = {k:v for k,v in metrics.items() if v.get('kind') != 'frozen_prior'}
    expected_modules = 130 if variant == 'action_only' else 256 if variant == 'conditioning_action' else 418
    if len(metrics) != 418 or len(solved) != expected_modules:
        raise ValueError('Incorrect calibrated scope')
    total = 0
    for key, value in solved.items():
        rows = 50 if key in ('model.time_mlp_in','model.time_mlp_out') else 800
        if value.get('rows_by_expert') != {name:rows for name in names}:
            raise ValueError(f'Row mismatch: {key}: {value.get("rows_by_expert")}')
        if not math.isfinite(value['ridge']) or value['ridge'] <= 0:
            raise ValueError('Invalid ridge')
        total += rows*len(names)
    return total
