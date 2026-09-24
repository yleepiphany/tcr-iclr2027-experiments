import pytest
import torch

from block_calibration import (calibrate_group, capture_request_features,
                               expert_group_state, group_scope, selected_state)
from native_trace import NativeActionTrace


class TinyMoT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mixtures = torch.nn.ModuleDict({
            "video": torch.nn.Module(), "action": torch.nn.Module()})
        self.mixtures["video"].blocks = torch.nn.ModuleList([
            torch.nn.Linear(2, 2) for _ in range(2)])
        self.mixtures["action"].blocks = torch.nn.ModuleList([
            torch.nn.Linear(2, 2) for _ in range(2)])


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mot = TinyMoT()
        self.proprio_encoder = torch.nn.Linear(2, 2)


def test_only_the_current_native_block_is_swapped_and_restored_on_error():
    model = TinyModel()
    original = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    expert = {"mot": {name: tensor.clone() for name, tensor in model.mot.state_dict().items()},
              "proprio_encoder": {name: tensor.clone()
                                   for name, tensor in model.proprio_encoder.state_dict().items()}}
    expert["mot"]["mixtures.video.blocks.0.weight"].add_(2)
    assert group_scope("02-video-block-00") == ("mot", ("mixtures.video.blocks.0.",))
    with pytest.raises(RuntimeError, match="abort"):
        with expert_group_state(model, "02-video-block-00", expert):
            assert torch.equal(model.mot.mixtures["video"].blocks[0].weight,
                               expert["mot"]["mixtures.video.blocks.0.weight"])
            assert torch.equal(model.mot.mixtures["video"].blocks[1].weight,
                               original["mot.mixtures.video.blocks.1.weight"])
            assert set(selected_state(model, "02-video-block-00")) == {
                "mixtures.video.blocks.0.weight", "mixtures.video.blocks.0.bias"}
            raise RuntimeError("abort")
    assert all(torch.equal(value, original[name]) for name, value in model.state_dict().items())


class NativeToyMoT(torch.nn.Module):
    num_layers = 1

    def __init__(self):
        super().__init__()
        video = torch.nn.Module()
        action = torch.nn.Module()
        video_block = torch.nn.Module()
        action_block = torch.nn.Module()
        video_block.self_attn = torch.nn.Module()
        action_block.self_attn = torch.nn.Module()
        video_block.self_attn.q = torch.nn.Linear(3, 3)
        action_block.self_attn.q = torch.nn.Linear(7, 7)
        video.blocks = torch.nn.ModuleList([video_block])
        action.blocks = torch.nn.ModuleList([action_block])
        self.mixtures = torch.nn.ModuleDict({"video": video, "action": action})

    def prefill_video_cache(self, *, video_tokens, video_freqs, video_t_mod,
                            video_context_payload, video_attention_mask):
        k = self.mixtures["video"].blocks[0].self_attn.q(video_tokens)
        return [{"k": k, "v": k - 1}]


class NativeToyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mot = NativeToyMoT()
        self.proprio_encoder = torch.nn.Linear(8, 7)

    def _append_proprio_to_context(self, *, context, context_mask, proprio):
        return torch.cat((context, self.proprio_encoder(proprio).unsqueeze(1)), dim=1), \
            torch.ones(1, 2, dtype=torch.bool)

    def _build_mot_attention_mask(self, *, video_seq_len, action_seq_len,
                                  video_tokens_per_frame, device):
        return torch.ones(video_seq_len + action_seq_len,
                          video_seq_len + action_seq_len, dtype=torch.bool)

    def _predict_action_noise_with_cache(self, *, latents_action, timestep_action,
                                         context, context_mask, video_kv_cache,
                                         attention_mask, video_seq_len):
        q = self.mot.mixtures["action"].blocks[0].self_attn.q(latents_action)
        return q + timestep_action + context.mean() * .1 + video_kv_cache[0]["k"].mean() * .01

    def infer_action(self, *, prompt, input_image, proprio, action_horizon,
                     num_inference_steps, seed):
        context, context_mask = self._append_proprio_to_context(
            context=torch.ones(1, 1, 7), context_mask=torch.ones(1, 1, dtype=torch.bool),
            proprio=proprio)
        mask = self._build_mot_attention_mask(video_seq_len=8, action_seq_len=32,
                                               video_tokens_per_frame=8, device="cpu")
        cache = self.mot.prefill_video_cache(
            video_tokens=torch.ones(1, 8, 3), video_freqs=torch.ones(1),
            video_t_mod=torch.ones(1),
            video_context_payload={"context": context, "mask": context_mask},
            video_attention_mask=mask[:8, :8])
        latents = torch.zeros(1, 32, 7)
        for index in range(10):
            prediction = self._predict_action_noise_with_cache(
                latents_action=latents, timestep_action=torch.tensor([index / 10]),
                context=context, context_mask=context_mask,
                video_kv_cache=cache, attention_mask=mask, video_seq_len=8)
            latents = latents + prediction * .1
        return {"action": latents[0]}


def test_block_solver_uses_native_expert_states_and_keeps_other_groups_fixed():
    torch.manual_seed(3)
    model = NativeToyPolicy()
    request = {"prompt": "move", "input_image": torch.zeros(1, 3, 224, 448),
               "proprio": torch.zeros(1, 8), "action_horizon": 32,
               "num_inference_steps": 10, "seed": 7}
    with NativeActionTrace(model) as trace:
        action = model.infer_action(**request)["action"]
    record = {"id": "A/spatial/task-00/request-000000",
              "native_request": request, "trace": trace.payload(native_action=action)}
    descriptor = {"module": "mot.mixtures.action.blocks.0.self_attn.q",
                  "weight_shape": [7, 7],
                  "native_calls_per_selected_request": 3}
    features, receipts = capture_request_features(
        model, [descriptor], record, seed=4, cap=6, device="cpu")
    assert features[descriptor["module"]].shape == (6, 8)  # augmented bias
    assert [len(row["selected"]) for row in receipts[descriptor["module"]]] == [2, 2, 2]
    original = {name: value.clone() for name, value in model.state_dict().items()}
    experts = {}
    requests = {}
    for ordinal, name in enumerate(("spatial", "object", "goal", "long")):
        mot = {key: value.clone() for key, value in model.mot.state_dict().items()}
        mot["mixtures.action.blocks.0.self_attn.q.weight"].add_(.03 * (ordinal + 1))
        experts[name] = {"mot": mot, "proprio_encoder": {
            key: value.clone() for key, value in model.proprio_encoder.state_dict().items()}}
        requests[name] = [{**record, "id": f"A/{name}/task-00/request-000000"}]
    result = calibrate_group(model, "04-action-block-00", [descriptor], experts,
                             requests, mass_rule="uniform", seed=4, cap=6,
                             device="cpu")
    assert result["complete"] and result["rows"] == 24
    assert not torch.equal(model.state_dict()[descriptor["module"] + ".weight"],
                           original[descriptor["module"] + ".weight"])
    assert all(torch.equal(value, original[key]) for key, value in model.state_dict().items()
               if key not in {descriptor["module"] + ".weight",
                              descriptor["module"] + ".bias"})
