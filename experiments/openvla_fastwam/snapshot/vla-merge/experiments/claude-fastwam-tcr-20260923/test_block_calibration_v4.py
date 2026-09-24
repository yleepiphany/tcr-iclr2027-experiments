import torch

import block_calibration_v4 as block


class AliasedPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.video_expert = torch.nn.Module()
        self.video_expert.text_embedding = torch.nn.Sequential(torch.nn.Linear(2, 2))
        self.mot = torch.nn.Module()
        self.mot.mixtures = torch.nn.ModuleDict({"video": self.video_expert})


def test_native_alias_capture_uses_frozen_mot_path(monkeypatch):
    model = AliasedPolicy()
    name = "mot.mixtures.video.text_embedding.0"
    assert name not in dict(model.named_modules())

    def replay(policy, _request, _trace, *, device):
        assert device == "cpu"
        policy.mot.mixtures["video"].text_embedding[0](torch.ones(5, 2))

    monkeypatch.setattr(block, "replay_selected_expert_states", replay)
    features, receipts = block.capture_request_features(
        model, [{"module": name, "weight_shape": [2, 2],
                 "native_calls_per_selected_request": 1}],
        {"id": "A/spatial/task-00/request-000000",
         "native_request": {}, "trace": {}}, seed=2, cap=3, device="cpu")
    assert features[name].shape == (3, 3)
    assert receipts[name][0]["available"] == 5
    assert len(receipts[name][0]["selected"]) == 3
