"""Read-only native OFT loader and exact request replay for TCR calibration.

Does not call the official get_vla(), which edits local checkpoint source/config.
No approximation of the action head, token layout, or normalization is introduced.
"""
from pathlib import Path
import copy
import json
import numpy as np


def configure_runtime():
    # TensorFlow is used for CPU image preprocessing only, never owns GPU memory.
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")


class NativePolicy:
    def __init__(self, checkpoint, suite):
        configure_runtime()
        import torch
        from experiments.robot.libero import run_libero_eval as native
        from experiments.robot import openvla_utils as utils
        self.torch, self.native, self.utils = torch, native, utils
        self.cfg = native.GenerateConfig(
            pretrained_checkpoint=str(Path(checkpoint).resolve()), task_suite_name=suite,
            unnorm_key=suite, use_l1_regression=True, use_diffusion=False,
            use_film=False, num_images_in_input=2, use_proprio=True,
            center_crop=True, num_open_loop_steps=8, use_wandb=False,
        )
        # Use the explicitly imported, recorded source classes, not mutable cached
        # remote-code copies. from_pretrained does not edit checkpoint contents.
        config = utils.OpenVLAConfig.from_pretrained(checkpoint, local_files_only=True)
        self.model, loading = utils.OpenVLAForActionPrediction.from_pretrained(
            checkpoint, config=config, torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True, local_files_only=True, output_loading_info=True,
        )
        if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise ValueError(f"Non-exact checkpoint load: {loading}")
        self.model = self.model.eval().to(utils.DEVICE)
        self.model.vision_backbone.set_num_images_in_input(2)
        self.model.norm_stats = json.loads((Path(checkpoint) / "dataset_statistics.json").read_text())
        native.check_unnorm_key(self.cfg, self.model)
        self.processor = load_processor(checkpoint)
        self.head = utils.get_action_head(self.cfg, self.model.llm_dim)
        self.proprio = utils.get_proprio_projector(self.cfg, self.model.llm_dim, proprio_dim=8)
        self.resize_size = native.get_image_resize_size(self.cfg)
        if native.NUM_ACTIONS_CHUNK != 8:
            raise ValueError("Wrong dataset constants: expected native OFT LIBERO chunk8")

    def request(self, observation, instruction, *, capture=False):
        captured = {}
        original = self.model.predict_action
        def record(*args, **kwargs):
            if args or captured:
                raise ValueError("Expected exactly one keyword-only native prediction")
            for key, value in kwargs.items():
                if key in {"action_head", "proprio_projector", "noisy_action_projector"}:
                    continue
                captured[key] = (value.detach().cpu().clone() if self.torch.is_tensor(value)
                                 else copy.deepcopy(value))
            return original(**kwargs)
        if capture:
            self.model.predict_action = record
        try:
            # Native helper normalizes proprio in-place: protect caller's raw input.
            actions = self.utils.get_vla_action(
                self.cfg, self.model, self.processor, copy.deepcopy(observation), instruction,
                action_head=self.head, proprio_projector=self.proprio,
            )
        finally:
            if capture:
                self.model.predict_action = original
        actions = validate_actions(actions)
        return (actions, captured) if capture else actions

    def replay(self, request):
        kwargs = {k: (v.to(self.utils.DEVICE) if self.torch.is_tensor(v) else copy.deepcopy(v))
                  for k, v in request.items()}
        with self.torch.inference_mode():
            actions, _ = self.model.predict_action(
                **kwargs, action_head=self.head, proprio_projector=self.proprio,
                noisy_action_projector=None,
            )
        return validate_actions(actions)

    def linear_modules(self):
        for prefix, root in (("backbone", self.model), ("action_head", self.head),
                             ("proprio_projector", self.proprio)):
            for name, module in root.named_modules():
                if isinstance(module, self.torch.nn.Linear):
                    yield f"{prefix}.{name}", module


def validate_actions(actions):
    values = np.asarray(actions)
    if values.shape != (8, 7) or not np.isfinite(values).all():
        raise ValueError(f"Invalid native actions: {values.shape}; expected finite (8,7)")
    return values


def load_processor(checkpoint):
    from transformers import AutoTokenizer
    from experiments.robot import openvla_utils as utils
    # Construct from the recorded source image class; AutoImageProcessor would
    # otherwise ask to import a different mutable checkpoint-local code copy.
    return utils.PrismaticProcessor(
        utils.PrismaticImageProcessor.from_pretrained(checkpoint, local_files_only=True),
        AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False),
    )
