"""Complete native OFT serialization, not proof that calibration was performed.

Keeps every backbone tensor (including unused pooling weights), both auxiliary
modules and unchanged preprocessing/normalization metadata. No checkpoint edits
in place, no expert-head routing, and no completion marker on partial failure.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from materialize_soup import SHARED, sha, write


def _snapshot(tensor):
    if not isinstance(tensor, torch.Tensor) or tensor.device.type == 'meta':
        raise ValueError('Only materialized tensors can be exported')
    value = tensor.detach().to('cpu').contiguous().clone()
    if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all():
        raise ValueError('Nonfinite exported tensor')
    return value


def export_native(states, template, run, *, provenance):
    """Serialize three complete state dicts using the prior's native file layout.

    `states` keys: backbone, action_head, proprio_projector. Their values are
    ordinary unprefixed native module state_dicts, never partial update dicts.
    Caller must freeze the model while this function writes it.
    """
    template, run = Path(template).resolve(), Path(run).resolve()
    if run.exists() or template == run or template in run.parents:
        raise FileExistsError('Export requires a new directory outside the source checkpoint')
    if set(states) != {'backbone', 'action_head', 'proprio_projector'}:
        raise ValueError('Complete backbone, action head and proprio projector required')
    json.dumps(provenance, allow_nan=False)
    index_path = template / 'model.safetensors.index.json'
    index = json.loads(index_path.read_text())
    mapping = index['weight_map']
    if set(states['backbone']) != set(mapping):
        raise ValueError('Partial or extra backbone state')
    if any(Path(name).name != name or not name.endswith('.safetensors') for name in mapping.values()):
        raise ValueError('Unsafe shard filename')
    component_keys = {}
    source_ids = {}
    paths = [index_path]
    metadata_names = [*SHARED, 'dataset_statistics.json']
    metadata_names += [name for name in ('tokenizer.model', 'added_tokens.json', 'generation_config.json')
                       if (template / name).exists()]
    paths += [template / name for name in metadata_names]
    for shard in sorted(set(mapping.values())):
        path = template / shard
        paths.append(path)
        with safe_open(str(path), framework='pt', device='cpu') as handle:
            expected = {key for key, value in mapping.items() if value == shard}
            if set(handle.keys()) != expected:
                raise ValueError('Template shard/index mismatch')
            for key in expected:
                if tuple(handle.get_slice(key).get_shape()) != tuple(states['backbone'][key].shape):
                    raise ValueError('Backbone tensor shape differs from template')
    for component in ('action_head', 'proprio_projector'):
        matches = list(template.glob(component + '--*_checkpoint.pt'))
        if len(matches) != 1:
            raise ValueError('Ambiguous template auxiliary checkpoint')
        paths += matches
        original = torch.load(str(matches[0]), map_location='cpu', weights_only=True, mmap=True)
        normalized = {key.removeprefix('module.'): key for key in original}
        if len(normalized) != len(original) or set(normalized) != set(states[component]):
            raise ValueError('Partial, duplicated or extra auxiliary state')
        if any(original[raw].shape != states[component][key].shape for key, raw in normalized.items()):
            raise ValueError('Auxiliary tensor shape differs from template')
        component_keys[component] = normalized
        del original
    for path in paths:
        stat = path.stat()
        source_ids[str(path)] = {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
                                'sha256': sha(path) if stat.st_size < 64 * 1024**2 else None}
    versions = [(value, value._version) for state in states.values() for value in state.values()]
    run.mkdir(parents=True, exist_ok=False)
    output = run / 'checkpoint'
    output.mkdir()
    write(run / 'started.json', {'serialization_only': True, 'template': str(template),
          'created_utc': datetime.now(timezone.utc).isoformat(), 'provenance': provenance,
          'source_identities': source_ids, 'source_sha256': sha(Path(__file__))})
    num_bytes, counts = 0, {}
    for shard in sorted(set(mapping.values())):
        values = {key: _snapshot(states['backbone'][key]) for key in sorted(mapping) if mapping[key] == shard}
        num_bytes += sum(value.numel() * value.element_size() for value in values.values())
        save_file(values, str(output / shard), metadata={'format': 'pt'})
        del values
    new_index = {**index, 'metadata': {**index.get('metadata', {}), 'total_size': num_bytes}}
    write(output / 'model.safetensors.index.json', new_index)
    for component, names in component_keys.items():
        values = {raw: _snapshot(states[component][key]) for key, raw in names.items()}
        torch.save(values, str(output / (component + '--tcr_checkpoint.pt')))
        counts[component] = len(values)
        del values
    for name in metadata_names:
        shutil.copyfile(template / name, output / name)
    if any(value._version != version for value, version in versions):
        raise ValueError('Live weights mutated during serialization')
    for name, record in source_ids.items():
        path = Path(name)
        stat = path.stat()
        if ((stat.st_size, stat.st_mtime_ns) != (record['size'], record['mtime_ns'])
                or (record['sha256'] is not None and sha(path) != record['sha256'])):
            raise ValueError('Template changed during serialization')
    result = {'complete': True, 'serialization_only': True, 'checkpoint': str(output),
              'backbone_keys': len(mapping), 'component_keys': counts,
              'files_sha256': {path.name: sha(path) for path in output.iterdir()},
              'provenance': provenance, 'gpu_reload_verified': False,
              'success_evaluated': False, 'calibration_completed_not_asserted_here': True}
    write(run / 'manifest.json', result)
    return result
