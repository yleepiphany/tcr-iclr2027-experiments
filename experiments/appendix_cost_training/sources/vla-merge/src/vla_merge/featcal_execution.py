"""Exact-input replay helpers for the FeatCal execution-source experiment."""
from __future__ import annotations

from collections import Counter
import torch


def validate_raw_manifest(manifest):
    if manifest.get("method") != "featcal_native_execution_input_capture":
        raise ValueError("Not a native execution capture")
    if manifest.get("demonstration_actions_used") is not False or manifest.get("success_filtering") is not False:
        raise ValueError("Capture source/selection contract differs")
    samples = manifest.get("samples", [])
    if manifest.get("sample_count") != 150 or len(samples) != 150:
        raise ValueError("Expected 150 native calls")
    expected = Counter((task, ordinal) for task in range(10) for ordinal in range(15))
    if Counter((x["task_slot"], x["task_ordinal"]) for x in samples) != expected:
        raise ValueError("Task/request quotas differ")
    if sorted(x["index"] for x in samples) != list(range(150)):
        raise ValueError("Sample indices must be unique and contiguous")
    for row in samples:
        if row["flow_index"] != (0, 5, 9)[row["task_ordinal"] % 3]:
            raise ValueError("Native flow mapping differs")
    for task in range(10):
        rows = [r for r in samples if r["task_slot"] == task]
        if len({r["prompt_signature"] for r in rows}) != 1:
            raise ValueError("Task stratum mixes prompts")
        for request in range(5):
            triple = [r for r in rows if r["task_ordinal"] // 3 == request]
            if len({r["request_index"] for r in triple}) != 1:
                raise ValueError("A flow triple mixes native requests")
    return samples


def stack_records(records, device):
    keys = set(records[0])
    if any(set(record) != keys for record in records):
        raise ValueError("Raw record keys differ within a batch")
    batch = {key: torch.cat([row[key] for row in records], dim=0).to(device) for key in keys}
    required = {"tokens", "masks", "x_t", "time", "native_velocity"}
    required |= {f"image_{i}" for i in range(3)} | {f"image_mask_{i}" for i in range(3)}
    if not required <= keys:
        raise ValueError("Native inputs incomplete")
    if batch["x_t"].shape[1:] != (50, 32) or batch["time"].shape != (len(records),):
        raise ValueError("Latent/time shape differs")
    if any(value.is_floating_point() and not torch.isfinite(value).all() for value in batch.values()):
        raise ValueError("Nonfinite native inputs")
    return batch


@torch.inference_mode()
def explicit_velocity_forward(policy, batch):
    """Full-joint graph used by the baseline, with *exact* recorded x_t and time.

Unlike model.forward's training interface, no demonstration actions, noise
interpolation or MSE are constructed. All calibrated modules follow the same
full-joint dependency graph. Native cached-prefix parity is measured separately.
"""
    from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks, prepare_attention_masks_4d

    model = policy.model
    prefix, prefix_pad, prefix_att = model.embed_prefix(
        [batch[f"image_{i}"] for i in range(3)],
        [batch[f"image_mask_{i}"] for i in range(3)],
        batch["tokens"], batch["masks"], batch.get("states"), batch.get("state_masks"),
    )
    suffix, suffix_pad, suffix_att, cond = model.embed_suffix(batch["x_t"], batch["time"])
    if model.paligemma_with_expert.paligemma.model.language_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
        prefix, suffix = prefix.to(torch.bfloat16), suffix.to(torch.bfloat16)
    pad = torch.cat([prefix_pad, suffix_pad], dim=1)
    att = torch.cat([prefix_att, suffix_att], dim=1)
    positions = torch.cumsum(pad, dim=1) - 1
    mask = prepare_attention_masks_4d(make_att_2d_masks(pad, att))
    (_, out), _ = model.paligemma_with_expert.forward(
        attention_mask=mask, position_ids=positions, past_key_values=None,
        inputs_embeds=[prefix, suffix], use_cache=False, adarms_cond=[None, cond],
    )
    velocity = model.action_out_proj(out[:, -model.config.chunk_size:].to(torch.float32))
    if not torch.isfinite(velocity).all():
        raise ValueError("Nonfinite replay velocity")
    return velocity
