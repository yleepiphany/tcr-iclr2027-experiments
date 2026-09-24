"""Identity gates compare the same graph, batch shape, and numerical precision."""
import torch


def validate_execution_identity(native_single, recorded_native, full_joint, baseline_oracle):
    for value in (native_single, recorded_native, full_joint, baseline_oracle):
        if not torch.isfinite(value).all():
            raise ValueError("Nonfinite identity-gate input")
    if native_single.shape != recorded_native.shape or native_single.dtype != recorded_native.dtype:
        raise ValueError("Native record shape/dtype mismatch")
    if full_joint.shape != baseline_oracle.shape or full_joint.dtype != baseline_oracle.dtype:
        raise ValueError("Baseline replay shape/dtype mismatch")
    if not torch.equal(native_single, recorded_native):
        raise ValueError("Native batch1 replay differs from recorded expert velocity")
    if not torch.equal(full_joint, baseline_oracle):
        raise ValueError("Execution adapter differs from original FeatCal full-joint graph")
    return {"native_batch1_bitwise_equal": True, "baseline_graph_bitwise_equal": True}
