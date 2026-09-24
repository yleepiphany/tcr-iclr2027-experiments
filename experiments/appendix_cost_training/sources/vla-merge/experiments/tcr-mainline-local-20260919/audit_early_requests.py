"""Audit first-five native requests against the complete execution count."""
from collections import defaultdict


def check_early(manifest: dict) -> dict:
    issues = []
    if manifest.get("request_mode") != "initial" or manifest.get("max_calls_per_prompt") != 1:
        issues.append("capture mode or episode cap differs")
    groups = defaultdict(list)
    for sample in manifest.get("samples", []):
        groups[str(sample.get("prompt_signature"))].append(sample)
    counts = manifest.get("prompt_seen_call_counts") or {}
    if set(groups) != set(counts):
        issues.append("prompt identities differ from full-call counts")
    for prompt, rows in groups.items():
        calls = counts.get(prompt)
        if type(calls) is not int or calls % 10 or not 50 <= calls < 1280:
            issues.append(f"{prompt}: invalid complete ten-step request count")
            continue
        by_slot = defaultdict(list)
        for row in rows:
            if row.get("task_episode_index") != 0:
                issues.append(f"{prompt}: noninitial episode")
            by_slot[row.get("selected_request_slot")].append(
                (row.get("request_index"), row.get("flow_index")))
        expected = {slot: [(slot, flow) for flow in (0, 5, 9)] for slot in range(5)}
        if {slot: sorted(values) for slot, values in by_slot.items()} != expected:
            issues.append(f"{prompt}: selected samples are not the first five complete requests")
        if manifest.get("selected_requests", {}).get(prompt) != list(range(5)):
            issues.append(f"{prompt}: selected-request provenance differs")
    return {"accepted": not issues, "issues": issues,
            "rule": "first five native requests from each complete, unscreened episode"}
