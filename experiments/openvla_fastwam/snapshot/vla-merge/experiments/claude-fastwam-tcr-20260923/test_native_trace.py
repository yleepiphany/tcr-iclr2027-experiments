import pytest
import torch

from native_trace import NativeActionTrace
from replay_native_trace import replay_native_trace


class ToyMoT:
    def prefill_video_cache(self, *, video_tokens, video_context_payload,
                            video_freqs=None, video_t_mod=None,
                            video_attention_mask=None):
        return [{"k": video_tokens + 1, "v": video_tokens - 1}]


class ToyPolicy:
    def __init__(self):
        self.mot = ToyMoT()

    def _predict_action_noise_with_cache(self, *, latents_action, timestep_action,
                                         context, context_mask, video_kv_cache,
                                         attention_mask, video_seq_len):
        assert video_seq_len == 2 and context_mask.all()
        return latents_action + timestep_action + video_kv_cache[0]["k"].mean() * 0.01

    def infer_action(self):
        cache = self.mot.prefill_video_cache(
            video_tokens=torch.ones(1, 2, 3),
            video_context_payload={"context": torch.ones(1, 2, 3), "mask": torch.ones(1, 2)})
        latents = torch.zeros(1, 2, 3)
        for step in range(10):
            prediction = self._predict_action_noise_with_cache(
                latents_action=latents, timestep_action=torch.tensor([step / 10]),
                context=torch.ones(1, 2, 3), context_mask=torch.ones(1, 2, dtype=torch.bool),
                video_kv_cache=cache, attention_mask=torch.ones(4, 4), video_seq_len=2)
            latents = latents + prediction * 0.1
        return latents


def test_real_call_order_and_actions_unchanged():
    policy = ToyPolicy()
    baseline = policy.infer_action()
    original_prefill = policy.mot.prefill_video_cache.__func__
    original_denoise = policy._predict_action_noise_with_cache.__func__
    with NativeActionTrace(policy) as trace:
        result = policy.infer_action()
    assert torch.equal(result, baseline)
    assert policy.mot.prefill_video_cache.__func__ is original_prefill
    assert policy._predict_action_noise_with_cache.__func__ is original_denoise
    assert "prefill_video_cache" not in policy.mot.__dict__
    assert "_predict_action_noise_with_cache" not in policy.__dict__
    payload = trace.payload(native_action=result)
    assert payload["selected_call_indices"] == [0, 5, 9]
    assert [float(row["inputs"]["timestep_action"][0]) for row in payload["calls"]] == pytest.approx([0, .5, .9])
    assert payload["video_cache_shapes"] == [{"k": (1, 2, 3), "v": (1, 2, 3)}]


def test_rejects_missing_denoising_call_and_restores_methods():
    policy = ToyPolicy()
    with pytest.raises(ValueError, match="incomplete"):
        with NativeActionTrace(policy):
            policy.mot.prefill_video_cache(video_tokens=torch.ones(1, 2, 3),
                                           video_context_payload={})
    assert policy.infer_action().shape == (1, 2, 3)


def test_rejects_cache_substitution_and_restores_methods():
    policy = ToyPolicy()
    with pytest.raises(ValueError, match="reuse"):
        with NativeActionTrace(policy):
            policy.mot.prefill_video_cache(video_tokens=torch.ones(1, 2, 3),
                                           video_context_payload={})
            policy._predict_action_noise_with_cache(
                latents_action=torch.ones(1, 2, 3), timestep_action=torch.zeros(1),
                context=torch.ones(1, 2, 3), context_mask=torch.ones(1, 2, dtype=torch.bool),
                video_kv_cache=[{"k": torch.ones(1, 2, 3), "v": torch.ones(1, 2, 3)}],
                attention_mask=torch.ones(4, 4), video_seq_len=2)
    assert policy.infer_action().shape == (1, 2, 3)


def test_native_prefill_and_selected_action_calls_replay_exactly():
    policy = ToyPolicy()
    with NativeActionTrace(policy) as trace:
        result = policy.infer_action()
    payload = trace.payload(native_action=result)
    # The toy prefill only uses two arguments; the real native prefill has five.
    # Supply the no-op native fields to test the production replay contract.
    payload["prefill_inputs"].update(
        video_freqs=torch.zeros(1), video_t_mod=torch.zeros(1),
        video_attention_mask=torch.ones(2, 2))
    assert replay_native_trace(policy, payload, device="cpu")["max_absolute_error"] == 0.0
    payload["calls"][1]["prediction"].add_(1)
    with pytest.raises(ValueError, match="parity failed at call 5"):
        replay_native_trace(policy, payload, device="cpu")


def test_nested_trace_rejected_without_changing_outer_capture():
    policy = ToyPolicy()
    with NativeActionTrace(policy) as trace:
        with pytest.raises(RuntimeError, match="nested"):
            with NativeActionTrace(policy):
                pass
        result = policy.infer_action()
    assert trace.payload(native_action=result)["expected_denoising_calls"] == 10
