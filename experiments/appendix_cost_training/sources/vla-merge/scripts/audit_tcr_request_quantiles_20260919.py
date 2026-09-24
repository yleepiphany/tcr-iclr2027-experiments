"""Check selected requests against the FULL native execution length, not a reservoir.

Applicable to the registered ten-denoising-step, one-episode-per-prompt captures.
Checks completed manifests only. Does not launch, alter or approve an experiment.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


def check_quantiles(manifest):
    issues, groups, details = [], defaultdict(list), {}
    if manifest.get('request_mode') != 'initial':
        issues.append('Capture mode is not registered initial/128 before across selection')
    if manifest.get('max_calls_per_prompt') != 1:
        issues.append('Capture episode cap differs from registered one episode per prompt')
    for sample in manifest['samples']:
        groups[str(sample['prompt_signature'])].append(sample)
    counts = manifest['prompt_seen_call_counts']
    if set(groups) != set(counts):
        issues.append('Prompt identities differ between samples and full-call counts')
    for prompt, rows in groups.items():
        calls = counts.get(prompt)
        if type(calls) is not int or calls % 10 or not 50 <= calls < 1280:
            issues.append(f'{prompt}: invalid ten-step full execution count/capture cap')
            continue
        total = calls // 10
        expected = [(total - 1) * slot // 4 for slot in range(5)]
        slots = defaultdict(list)
        for row in rows:
            if row['task_episode_index'] != 0:
                issues.append(f'{prompt}: more than the registered single episode')
            slots[row['selected_request_slot']].append((row['request_index'], row['flow_index']))
        expected_rows = {s: [(r, f) for f in (0, 5, 9)] for s, r in enumerate(expected)}
        observed_rows = {s: sorted(values) for s, values in slots.items()}
        if observed_rows != expected_rows:
            issues.append(f'{prompt}: slots are not quantiles of ALL {total} native requests')
        selected = manifest.get('selected_requests', {}).get(prompt)
        if selected != expected:
            issues.append(f'{prompt}: selected_requests provenance differs from full quantiles')
        details[prompt] = dict(total_native_requests=total, expected=expected,
                               recorded_selected_requests=selected)
    return dict(matches_registered_full_execution_quantiles=not issues,
                issues=issues, prompts=details,
                assumption='Ten denoising calls per native request, one complete episode per prompt; '
                           'native collector increments seen-call counters even for unretained requests.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    reports = {}
    for path in args.manifest:
        raw = path.read_bytes()
        reports[str(path)] = dict(manifest_sha256=hashlib.sha256(raw).hexdigest(),
                                 **check_quantiles(json.loads(raw)))
    with args.output.open('x') as stream:
        json.dump(reports, stream, indent=2)
        stream.write('\n')
    print(json.dumps({p: dict(matches=x['matches_registered_full_execution_quantiles'],
                             violations=len(x['issues'])) for p, x in reports.items()}, indent=2))
