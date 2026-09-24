import torch

from native_trace import NativeActionTrace
from replay_merged_prefix import replay_selected_expert_states


class VisualMoT:
    num_layers = 1

    def prefill_video_cache(self, *, video_tokens, video_freqs, video_t_mod,
                            video_context_payload, video_attention_mask):
        return [{"k": video_tokens + 1, "v": video_tokens - 1}]


class PrefixPolicy:
    def __init__(self):
        self.mot = VisualMoT()

    def _append_proprio_to_context(self, *, context, context_mask, proprio):
        return torch.cat((context, proprio[:, :7, None].transpose(1, 2)), dim=1), \
            torch.ones(1, 2, dtype=torch.bool)

    def _build_mot_attention_mask(self, *, video_seq_len, action_seq_len,
                                  video_tokens_per_frame, device):
        return torch.ones(video_seq_len + action_seq_len,
                          video_seq_len + action_seq_len, dtype=torch.bool)

    def _predict_action_noise_with_cache(self, *, latents_action, timestep_action,
                                         context, context_mask, video_kv_cache,
                                         attention_mask, video_seq_len):
        return latents_action + timestep_action + context.mean() * .1 + \
            video_kv_cache[0]["k"].mean() * .01

    def infer_action(self, *, prompt, input_image, proprio, action_horizon,
                     num_inference_steps, seed):
        context, context_mask = self._append_proprio_to_context(
            context=torch.ones(1, 1, 7),
            context_mask=torch.ones(1, 1, dtype=torch.bool), proprio=proprio)
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=2, action_seq_len=32,
            video_tokens_per_frame=2, device="cpu")
        cache = self.mot.prefill_video_cache(
            video_tokens=torch.ones(1, 2, 3), video_freqs=torch.ones(1),
            video_t_mod=torch.ones(1),
            video_context_payload={"context": context, "mask": context_mask},
            video_attention_mask=attention_mask[:2, :2])
        latents = torch.zeros(1, 32, 7)
        for index in range(10):
            prediction = self._predict_action_noise_with_cache(
                latents_action=latents, timestep_action=torch.tensor([index / 10]),
                context=context, context_mask=context_mask,
                video_kv_cache=cache, attention_mask=attention_mask,
                video_seq_len=2)
            latents = latents + prediction * .1
        return {"action": latents[0]}


def test_selected_expert_states_use_the_current_full_prefix():
    policy = PrefixPolicy()
    request = {"prompt": "move", "input_image": torch.zeros(1, 3, 224, 448),
               "proprio": torch.zeros(1, 8), "action_horizon": 32,
               "num_inference_steps": 10, "seed": 7}
    with NativeActionTrace(policy) as captured:
        action = policy.infer_action(**request)["action"]
    trace = captured.payload(native_action=action)
    predictions, prefix = replay_selected_expert_states(
        policy, request, trace, device="cpu")
    assert all(torch.equal(prediction, row["prediction"])
               for prediction, row in zip(predictions, trace["calls"]))
    assert len(prefix["video_kv_cache"]) == 1
    assert "prefill_video_cache" not in policy.mot.__dict__
    assert "_append_proprio_to_context" not in policy.__dict__
    changed_request = {**request, "proprio": torch.ones(1, 8)}
    changed, _ = replay_selected_expert_states(policy, changed_request, trace,
                                                device="cpu")
    assert all(not torch.equal(before, after)
               for before, after in zip(predictions, changed))
