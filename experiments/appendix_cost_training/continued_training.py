"""Report Tables6/15 protocol blockers; deliberately contains no training launch."""
import argparse
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ARMS=['common_base','single_expert_dev_selected','model_soups','regmean_pp','featcal','tcr']
REQUIRED=['budget_updates_B','source_dataset_revision_and_mixture','trainable_parameter_scope',
          'optimizer_and_scheduler','global_batch_and_accumulation','random_seeds',
          'evaluation_update_schedule','shared_success_threshold_tau','initialization_checkpoint_bindings',
          'single_expert_development_selection_rule','restart_and_optimizer_state_semantics']

def inspect(path):
    config=json.loads(Path(path).read_text())
    missing=[key for key in REQUIRED if config.get(key) is None]
    if config.get('arms')!=ARMS:missing.append('exact_six_arm_matrix')
    return {'status':'BLOCKED','tables':[6,15],'missing_protocol_fields':missing,
            'frozen_reviewed_protocol_exists':False,'training_started':False,
            'reason':'The paper contains an unfilled six-arm design; no reviewed shared training budget/configuration was found. A generic expert-training script is not a matched six-arm protocol.',
            'required_next_step':'Freeze and independently review the missing scientific/training choices before implementing an executable trainer. Filling this template alone is not approval or validation.'}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=HERE/'configs/continued_training.blocked.json')
    args=parser.parse_args();print(json.dumps(inspect(args.config),indent=2));raise SystemExit(2)

if __name__=='__main__':main()
