import pytest
import torch

from block_calibration_v3 import resolve_modules


class AliasedNativePolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.video_expert = torch.nn.Module()
        self.video_expert.text_embedding = torch.nn.Sequential(torch.nn.Linear(2, 2))
        self.action_expert = torch.nn.Module()
        self.action_expert.text_embedding = torch.nn.Sequential(torch.nn.Linear(2, 2))
        self.mot = torch.nn.Module()
        self.mot.mixtures = torch.nn.ModuleDict({
            "video": self.video_expert, "action": self.action_expert})


def test_frozen_mot_alias_paths_resolve_to_native_modules():
    model = AliasedNativePolicy()
    names = ["mot.mixtures.video.text_embedding.0",
             "mot.mixtures.action.text_embedding.0"]
    assert all(name not in dict(model.named_modules()) for name in names)
    found = resolve_modules(model, names)
    assert found[names[0]] is model.video_expert.text_embedding[0]
    assert found[names[1]] is model.action_expert.text_embedding[0]


def test_missing_path_and_same_module_twice_are_rejected():
    model = AliasedNativePolicy()
    with pytest.raises(ValueError, match="path absent"):
        resolve_modules(model, ["mot.mixtures.video.no_such_layer"])
    with pytest.raises(ValueError, match="alias the same"):
        resolve_modules(model, ["video_expert.text_embedding.0",
                                "mot.mixtures.video.text_embedding.0"])
