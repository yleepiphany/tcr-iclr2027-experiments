"""CPU-only action agreement for the current paper; not the old 7-D diagnostic.

The caller must freeze a common coordinate scale from training statistics (or
ones for already common-normalized actions). This module never estimates scales
from predictions and never evaluates success. Gripper/padded coordinates are not
part of the continuous MSE/cosine metric.
"""
import math
import torch


def continuous_metrics(prediction, teacher, *, coordinate_scale,
                       continuous_dims=(0, 1, 2, 3, 4, 5), execute_steps=10,
                       cosine_eps=1e-12):
    def shape(value):
        value = torch.as_tensor(value).detach().cpu().to(torch.float64)
        if value.ndim == 3 and value.shape[0] == 1:
            value = value[0]
        if value.ndim != 2 or value.shape[0] < execute_steps:
            raise ValueError("Expected one full native action chunk")
        if not torch.isfinite(value).all():
            raise ValueError("Non-finite actions, including outside the executed mask")
        return value
    if type(execute_steps) is not int or execute_steps <= 0:
        raise ValueError("Invalid executed-action count")
    if not math.isfinite(cosine_eps) or cosine_eps <= 0:
        raise ValueError("A fixed positive cosine norm threshold is required")
    dims = tuple(continuous_dims)
    if not dims or len(set(dims)) != len(dims) or any(type(d) is not int or d < 0 for d in dims):
        raise ValueError("Invalid continuous-action dimension mask")
    a, b = shape(prediction), shape(teacher)
    if a.shape != b.shape or max(dims) >= a.shape[1]:
        raise ValueError("Action shapes/mask differ")
    scale = torch.as_tensor(coordinate_scale, dtype=torch.float64).cpu()
    if scale.shape != (len(dims),) or not torch.isfinite(scale).all() or not (scale > 0).all():
        raise ValueError("Common scale must be a positive finite vector")
    a, b = (a[:execute_steps, dims] * scale).flatten(), (b[:execute_steps, dims] * scale).flatten()
    norm_a, norm_b = float(a.norm()), float(b.norm())
    valid = norm_a > cosine_eps and norm_b > cosine_eps
    cosine = max(-1., min(1., float(a.dot(b)) / (norm_a * norm_b))) if valid else None
    return {"mse": float((a - b).square().mean()), "cosine": cosine,
            "cosine_valid": bool(valid), "continuous_coordinates": a.numel(),
            "prediction_norm": norm_a, "teacher_norm": norm_b}


def task_macro(rows, *, expected_tasks, requests_per_task=10):
    if not rows or requests_per_task <= 0:
        raise ValueError("Empty or invalid measurement set")
    expected = set(expected_tasks)
    if len(expected) != len(expected_tasks):
        raise ValueError("Duplicate expected task identity")
    grouped, seen = {}, set()
    for row in rows:
        key = (row['suite'], row['task_id'])
        identity = (key, row['episode_id'], row['request_id'])
        if identity in seen or key not in expected:
            raise ValueError("Duplicate request or unexpected task")
        seen.add(identity)
        if not math.isfinite(row['mse']) or row['mse'] < 0:
            raise ValueError("Invalid request MSE")
        if type(row['cosine_valid']) is not bool or row['cosine_valid'] != (row['cosine'] is not None):
            raise ValueError("Cosine coverage flag/value mismatch")
        if row['cosine'] is not None and (not math.isfinite(row['cosine']) or not -1 <= row['cosine'] <= 1):
            raise ValueError("Invalid cosine")
        grouped.setdefault(key, []).append(row)
    if set(grouped) != expected or any(len(v) != requests_per_task for v in grouped.values()):
        raise ValueError("Incomplete task/request coverage")
    task_rows = []
    for (suite, task_id), values in sorted(grouped.items()):
        valid = [v['cosine'] for v in values if v['cosine_valid']]
        task_rows.append({'suite': suite, 'task_id': task_id,
                          'mse': sum(v['mse'] for v in values) / len(values),
                          'cosine': sum(valid) / len(valid) if valid else None,
                          'requests': len(values), 'cosine_valid': len(valid)})
    complete_cosine = all(t['cosine'] is not None for t in task_rows)
    # A task with no valid cosine is not silently dropped from a task-macro score.
    return {'mse': sum(t['mse'] for t in task_rows) / len(task_rows),
            'cosine': sum(t['cosine'] for t in task_rows) / len(task_rows) if complete_cosine else None,
            'cosine_tasks_complete': complete_cosine, 'requests': len(rows),
            'cosine_valid_requests': sum(t['cosine_valid'] for t in task_rows),
            'cosine_valid_fraction': sum(t['cosine_valid'] for t in task_rows) / len(rows),
            'per_task': task_rows}
