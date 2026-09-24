import copy
import pytest
import torch
from pi05_solver_bridge import REFERENCE, solve_module


@pytest.mark.parametrize('mass_rule', ['relative', 'uniform'])
@pytest.mark.parametrize('bias', [True, False])
@pytest.mark.parametrize('ridge', [None, .123])
def test_exact_main_solver(mass_rule, bias, ridge):
    torch.manual_seed(71)
    module = torch.nn.Linear(5, 3, bias=bias)
    prior = module.weight.detach().clone()
    if bias:
        prior = torch.cat((prior, module.bias.detach()[:, None]), dim=1)
    xs = [torch.randn(11 + i, prior.shape[1]) for i in range(4)]
    if bias:
        for x in xs:
            x[:, -1] = 1
    ws = [prior + .3 * torch.randn_like(prior) for _ in range(4)]
    names = ('spatial', 'object', 'goal', 'long')
    expected, report = REFERENCE(dict(zip(names, xs)), dict(zip(names, ws)), prior,
        ridge_ratio=.05, ridge_scale='feature_energy', max_correction_ratio=3.,
        expert_loss_normalization='prior' if mass_rule == 'relative' else 'none', fixed_ridge=ridge)
    actual = solve_module(module, xs, ws, prior, mass_rule=mass_rule, ridge_multiplier=.05,
        max_correction_ratio=3., fixed_ridge=ridge,
        ridge_source='computed_pass_A' if ridge is None else 'fixed_from_same_candidate_pass_A')
    assert torch.equal(module.weight, expected[:, :5])
    if bias:
        assert torch.equal(module.bias, expected[:, -1])
    assert actual['ridge'] == report['ridge']
    assert actual['trust_scale'] == report['trust_scale']


def test_fixed_ridge_and_invalid_input():
    module = torch.nn.Linear(2, 1, bias=False)
    prior = module.weight.detach().clone()
    x = [torch.randn(6, 2) for _ in range(4)]
    weights = [prior + 1 for _ in range(4)]
    for scale in [1., 10.]:
        result = solve_module(copy.deepcopy(module), [scale*t for t in x], weights, prior,
            mass_rule='uniform', ridge_multiplier=.05, max_correction_ratio=3.,
            fixed_ridge=.123, ridge_source='fixed_from_same_candidate_pass_A')
        assert result['ridge'] == .123
    with pytest.raises(ValueError):
        solve_module(module, x, weights, prior, mass_rule='uniform', ridge_multiplier=.05,
            max_correction_ratio=3., fixed_ridge=-1)
