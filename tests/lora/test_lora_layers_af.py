# coding=utf-8
# Copyright 2024 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import sys
import unittest

import torch
from transformers import AutoTokenizer, T5EncoderModel

from diffusers import (
    AuraFlowPipeline,
    AuraFlowTransformer2DModel,
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.utils.testing_utils import (
    floats_tensor,
    is_peft_available,
    require_peft_backend,
)


if is_peft_available():
    pass

sys.path.append(".")

from utils import PeftLoraLoaderMixinTests  # noqa: E402


@require_peft_backend
class AuraFlowLoRATests(unittest.TestCase, PeftLoraLoaderMixinTests):
    pipeline_class = AuraFlowPipeline
    scheduler_cls = FlowMatchEulerDiscreteScheduler()
    scheduler_kwargs = {}
    uses_flow_matching = True
    transformer_kwargs = {
        "sample_size": 64,
        "patch_size": 1,
        "in_channels": 4,
        "num_mmdit_layers": 1,
        "num_single_dit_layers": 1,
        "attention_head_dim": 16,
        "num_attention_heads": 2,
        "joint_attention_dim": 32,
        "caption_projection_dim": 32,
        "out_channels": 4,
        "pos_embed_max_size": 64,
    }
    transformer_cls = AuraFlowTransformer2DModel
    tokenizer_cls, tokenizer_id = AutoTokenizer, "hf-internal-testing/tiny-random-t5"
    text_encoder_cls, text_encoder_id = T5EncoderModel, "hf-internal-testing/tiny-random-t5"

    vae_kwargs = {
        "sample_size": 32,
        "in_channels": 3,
        "out_channels": 3,
        "block_out_channels": (4,),
        "layers_per_block": 1,
        "latent_channels": 4,
        "norm_num_groups": 1,
        "use_quant_conv": False,
        "use_post_quant_conv": False,
        "shift_factor": None,
        "scaling_factor": 0.13025,
    }

    @property
    def output_shape(self):
        return (1, 64, 64, 3)

    def get_dummy_inputs(self, with_generator=True):
        batch_size = 1
        sequence_length = 10
        num_channels = 4
        sizes = (32, 32)

        generator = torch.manual_seed(0)
        noise = floats_tensor((batch_size, num_channels) + sizes)
        input_ids = torch.randint(1, sequence_length, size=(batch_size, sequence_length), generator=generator)

        pipeline_inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "num_inference_steps": 4,
            "guidance_scale": 0.0,
            "height": 8,
            "width": 8,
            "output_type": "np",
        }
        if with_generator:
            pipeline_inputs.update({"generator": generator})

        return noise, input_ids, pipeline_inputs
