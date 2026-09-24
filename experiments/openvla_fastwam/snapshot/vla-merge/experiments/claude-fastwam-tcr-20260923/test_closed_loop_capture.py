from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from capture_closed_loop import ClosedLoopEpisodeCapture, request_quantiles
from replay_native_trace import replay_native_trace
from test_native_trace import ToyPolicy


class ActionPolicy(ToyPolicy):
    def infer_action(self, **kwargs):
        assert torch.equal(kwargs["input_image"], torch.zeros(1))
        return {"action": super().infer_action()}


def test_native_episode_capture_binds_actual_observations_and_executed_chunks(tmp_path: Path):
    policy = ActionPolicy()

    def native_chunk(*, obs, model):
        action = model.infer_action(input_image=torch.zeros(1))["action"][0].numpy()
        return action, {"image": obs["image"]}, None

    native = SimpleNamespace(_predict_action_chunk=native_chunk)
    with ClosedLoopEpisodeCapture(native, policy, tmp_path / "episode-0",
                                  {"suite": "spatial", "task_id": 0}) as capture:
        outputs = [native._predict_action_chunk(obs={"image": np.full((2, 2), i, np.uint8)},
                                                model=policy)[0] for i in range(3)]
    receipt = capture.complete(initial_state=np.array([1.0, 2.0]), success=False)
    assert receipt["selected_request_indices"] == [0, 1, 2]
    assert receipt["success"] is False
    assert native._predict_action_chunk is native_chunk
    assert "infer_action" not in policy.__dict__
    rows = [torch.load(tmp_path / "episode-0" / f"request-{i:06d}.pt",
                       weights_only=True) for i in range(3)]
    assert len({r["raw_observation_sha256"] for r in rows}) == 3
    for row, action in zip(rows, outputs):
        from capture_closed_loop import observation_sha256
        assert observation_sha256(row["raw_observation"]) == row["raw_observation_sha256"]
        assert np.array_equal(row["executed_action_chunk"].numpy(), action)
        assert replay_native_trace(policy, _full_prefill_contract(row["trace"]),
                                   device="cpu")["accepted"]


def _full_prefill_contract(payload):
    # The toy MoT accepts only two inputs; the actual native MoT accepts five.
    payload["prefill_inputs"].update(
        video_freqs=torch.zeros(1), video_t_mod=torch.zeros(1),
        video_attention_mask=torch.ones(2, 2))
    return payload


def test_capture_rejects_unpaired_native_request_and_restores_methods(tmp_path: Path):
    policy = ActionPolicy()
    native = SimpleNamespace(_predict_action_chunk=lambda *, obs, model: (
        np.zeros((2, 3)), {}, None))
    original = native._predict_action_chunk
    with pytest.raises(ValueError, match="exactly one action inference"):
        with ClosedLoopEpisodeCapture(native, policy, tmp_path / "episode-1", {}) as capture:
            native._predict_action_chunk(obs={"image": np.zeros((2, 2))}, model=policy)
    assert native._predict_action_chunk is original
    assert "infer_action" not in policy.__dict__
    assert request_quantiles(1) == [0]
    assert request_quantiles(2) == [0, 1]
