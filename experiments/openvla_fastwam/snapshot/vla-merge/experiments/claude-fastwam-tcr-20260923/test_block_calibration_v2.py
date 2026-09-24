import torch

import block_calibration_v2 as block


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proprio_encoder = torch.nn.Linear(2, 2)
        self.mot = torch.nn.Linear(2, 2)


def test_root_proprio_linear_is_recognized_as_solved(monkeypatch):
    model = TinyModel()
    original_mot = {key: value.clone() for key, value in model.mot.state_dict().items()}
    names = ("spatial", "object", "goal", "long")
    experts = {}
    requests = {}
    for index, name in enumerate(names):
        state = {key: value.clone() for key, value in model.proprio_encoder.state_dict().items()}
        state["weight"].add_(0.1 * (index + 1))
        experts[name] = {"proprio_encoder": state}
        requests[name] = [{"id": f"A/{name}/task00/request00"}]

    def capture(_model, _descriptors, _record, **_kwargs):
        return {"proprio_encoder": torch.tensor([
            [1.0, 0.0, 1.0], [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0], [2.0, 1.0, 1.0]])}, {}

    monkeypatch.setattr(block, "capture_request_features", capture)
    result = block.calibrate_group(
        model, "00-proprio", [{"module": "proprio_encoder"}], experts, requests,
        mass_rule="uniform", seed=1, cap=4, device="cpu")
    assert result["complete"] is True
    assert result["rows"] == 16
    assert all(torch.equal(model.mot.state_dict()[key], value)
               for key, value in original_mot.items())
