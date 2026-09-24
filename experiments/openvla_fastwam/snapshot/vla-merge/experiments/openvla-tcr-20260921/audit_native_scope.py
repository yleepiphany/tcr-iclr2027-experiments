"""CPU/meta audit: separate registered parameters from the native action path."""
import argparse
import json
from pathlib import Path

import torch
from accelerate import init_empty_weights

from expert_bank import ExpertBank
from materialize_soup import ORDER, sha, write
from native_oft import configure_runtime


def main():
    p = argparse.ArgumentParser()
    for name in ('checkpoint', 'ledger', 'block-plan', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    args = p.parse_args()
    configure_runtime()
    from experiments.robot import openvla_utils as u
    config = u.OpenVLAConfig.from_pretrained(args.checkpoint, local_files_only=True)
    with init_empty_weights():
        model = u.OpenVLAForActionPrediction(config)
        head = u.L1RegressionActionHead(input_dim=model.llm_dim, hidden_dim=model.llm_dim, action_dim=7)
        proprio = u.ProprioProjector(llm_dim=model.llm_dim, proprio_dim=8)
    registered = {prefix + '.' + name: module for prefix, root in (
        ('backbone', model), ('action_head', head), ('proprio_projector', proprio))
        for name, module in root.named_modules() if isinstance(module, torch.nn.Linear)}
    plan = json.loads(args.block_plan.read_text())
    executed = {row['module'] for rows in plan['blocks'].values() for row in rows}
    if executed - registered.keys():
        raise ValueError('Trace contains unregistered Linear')
    missing = sorted(registered.keys() - executed)
    details = []
    with ExpertBank(args.ledger) as bank:
        for name in missing:
            parameters = []
            for key, current in registered[name].state_dict().items():
                values = [bank.tensor(expert, name + '.' + key) for expert in ORDER]
                if any(value.shape != current.shape for value in values):
                    raise ValueError('Source shape mismatch')
                parameters.append({'tensor': name + '.' + key, 'shape': list(current.shape),
                                   'identical_across_experts': all(torch.equal(values[0], value) for value in values[1:])})
            details.append({'module': name, 'parameters': parameters})
    result = {'registered_linears': len(registered), 'observed_linears': len(executed),
              'registered_but_not_observed': details,
              'block_plan_sha256': sha(args.block_plan), 'ledger_sha256': sha(args.ledger),
              'meta_only': True, 'gpu_used': False,
              'does_not_prove_unreachability': True,
              'interpretation': 'These modules require source-path inspection; no silent removal from export scope.'}
    write(args.output, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
