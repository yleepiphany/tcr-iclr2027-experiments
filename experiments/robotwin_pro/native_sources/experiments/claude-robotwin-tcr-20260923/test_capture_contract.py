import pytest
import torch

import capture_contract as contract
from run_capture import resolve_native_policy, resolve_noise_generator


def rows(requests):
    metadata = []
    values = []
    for request in range(requests):
        for flow in contract.FLOW_INDICES:
            metadata.append({"request_index": request, "flow_index": flow})
            values.append((request, flow))
    return values, metadata


def test_exact_full_trajectory_quantiles_and_flows():
    values, metadata = rows(9)
    chosen, selected, provenance = contract.select_request_quantiles(values, metadata)
    assert provenance == {"available_request_count": 9, "selected_request_indices": [0, 2, 4, 6, 8]}
    assert chosen == [(request, flow) for request in (0, 2, 4, 6, 8) for flow in (0, 5, 9)]
    assert [row["selected_request_slot"] for row in selected] == [slot for slot in range(5) for _ in range(3)]


def test_short_trajectory_is_rejected_without_duplication():
    values, metadata = rows(4)
    with pytest.raises(ValueError, match="at least five"):
        contract.select_request_quantiles(values, metadata)


def test_incomplete_flow_group_is_rejected():
    values, metadata = rows(5)
    values.pop(); metadata.pop()
    with pytest.raises(ValueError, match="flow 0/5/9"):
        contract.select_request_quantiles(values, metadata)


def test_seed_derivation_skips_identity_collision_only():
    first = contract.uint31(contract.SEED_NAMESPACE, "reset", 1, "A", 8, 0)
    occupied = {(8, first)}
    seed, counter = contract.derive_unique_seed(8, 1, "A", occupied)
    assert counter == 1 and seed != first and (8, seed) in occupied


def test_noise_generator_resolves_through_policy_wrappers():
    PI05Pytorch = type(
        "PI05Pytorch",
        (torch.nn.Module,),
        {"sample_noise": lambda self, shape, device: None},
    )

    class PI05Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = PI05Pytorch()

        def predict_action_chunk(self, batch):
            return batch

        def reset(self):
            return None

    class PeftWrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = PI05Policy()

    wrapped = PeftWrapper()
    assert resolve_native_policy(wrapped) is wrapped.model
    assert resolve_noise_generator(wrapped) is wrapped.model.model


def test_noise_generator_rejects_ambiguous_or_missing_core():
    PI05Pytorch = type(
        "PI05Pytorch",
        (torch.nn.Module,),
        {"sample_noise": lambda self, shape, device: None},
    )
    with pytest.raises(RuntimeError, match="found 0"):
        resolve_noise_generator(torch.nn.Module())
    pair = torch.nn.ModuleList([PI05Pytorch(), PI05Pytorch()])
    with pytest.raises(RuntimeError, match="found 2"):
        resolve_noise_generator(pair)


def test_native_policy_rejects_ambiguous_or_missing_interface():
    PI05Policy = type(
        "PI05Policy",
        (torch.nn.Module,),
        {"predict_action_chunk": lambda self, batch: None, "reset": lambda self: None},
    )
    with pytest.raises(RuntimeError, match="found 0"):
        resolve_native_policy(torch.nn.Module())
    pair = torch.nn.ModuleList([PI05Policy(), PI05Policy()])
    with pytest.raises(RuntimeError, match="found 2"):
        resolve_native_policy(pair)
