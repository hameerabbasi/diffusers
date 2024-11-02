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
import gc
import sys
import unittest

import numpy as np
import pytest
import torch
from transformers import AutoTokenizer, CLIPTextModelWithProjection, CLIPTokenizer, T5EncoderModel

from diffusers import (
    FlowMatchEulerDiscreteScheduler,
    SD3Transformer2DModel,
    StableDiffusion3Img2ImgPipeline,
    StableDiffusion3Pipeline,
)
from diffusers.utils import load_image
from diffusers.utils.import_utils import is_accelerate_available
from diffusers.utils.testing_utils import (
    is_peft_available,
    nightly,
    numpy_cosine_similarity_distance,
    print_tensor_test,
    require_big_gpu_with_torch_cuda,
    require_peft_backend,
    require_torch_gpu,
    slow,
    torch_device,
)


if is_peft_available():
    pass

sys.path.append(".")

from utils import PeftLoraLoaderMixinTests  # noqa: E402


if is_accelerate_available():
    from accelerate.utils import release_memory


@require_peft_backend
class SD3LoRATests(unittest.TestCase, PeftLoraLoaderMixinTests):
    pipeline_class = StableDiffusion3Pipeline
    scheduler_cls = FlowMatchEulerDiscreteScheduler
    scheduler_kwargs = {}
    scheduler_classes = [FlowMatchEulerDiscreteScheduler]
    transformer_kwargs = {
        "sample_size": 32,
        "patch_size": 1,
        "in_channels": 4,
        "num_layers": 1,
        "attention_head_dim": 8,
        "num_attention_heads": 4,
        "caption_projection_dim": 32,
        "joint_attention_dim": 32,
        "pooled_projection_dim": 64,
        "out_channels": 4,
    }
    transformer_cls = SD3Transformer2DModel
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
        "shift_factor": 0.0609,
        "scaling_factor": 1.5035,
    }
    has_three_text_encoders = True
    tokenizer_cls, tokenizer_id = CLIPTokenizer, "hf-internal-testing/tiny-random-clip"
    tokenizer_2_cls, tokenizer_2_id = CLIPTokenizer, "hf-internal-testing/tiny-random-clip"
    tokenizer_3_cls, tokenizer_3_id = AutoTokenizer, "hf-internal-testing/tiny-random-t5"
    text_encoder_cls, text_encoder_id = CLIPTextModelWithProjection, "hf-internal-testing/tiny-sd3-text_encoder"
    text_encoder_2_cls, text_encoder_2_id = CLIPTextModelWithProjection, "hf-internal-testing/tiny-sd3-text_encoder-2"
    text_encoder_3_cls, text_encoder_3_id = T5EncoderModel, "hf-internal-testing/tiny-random-t5"

    @property
    def output_shape(self):
        return (1, 32, 32, 3)

    @require_torch_gpu
    def test_sd3_lora(self):
        """
        Test loading the loras that are saved with the diffusers and peft formats.
        Related PR: https://github.com/huggingface/diffusers/pull/8584
        """
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components[0])
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        lora_model_id = "hf-internal-testing/tiny-sd3-loras"

        lora_filename = "lora_diffusers_format.safetensors"
        pipe.load_lora_weights(lora_model_id, weight_name=lora_filename)
        pipe.unload_lora_weights()

        lora_filename = "lora_peft_format.safetensors"
        pipe.load_lora_weights(lora_model_id, weight_name=lora_filename)

    @unittest.skip("Not supported in SD3.")
    def test_simple_inference_with_text_denoiser_block_scale(self):
        pass

    @unittest.skip("Not supported in SD3.")
    def test_simple_inference_with_text_denoiser_multi_adapter_block_lora(self):
        pass

    @unittest.skip("Not supported in SD3.")
    def test_simple_inference_with_text_denoiser_block_scale_for_all_dict_options(self):
        pass

    @unittest.skip("Not supported in SD3.")
    def test_modify_padding_mode(self):
        pass


@slow
@nightly
@require_torch_gpu
@require_peft_backend
@require_big_gpu_with_torch_cuda
@pytest.mark.big_gpu_with_torch_cuda
class SD3LoraIntegrationTests(unittest.TestCase):
    pipeline_class = StableDiffusion3Img2ImgPipeline
    repo_id = "stabilityai/stable-diffusion-3-medium-diffusers"

    def setUp(self):
        super().setUp()
        gc.collect()
        torch.cuda.empty_cache()

    def tearDown(self):
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def get_inputs(self, device, seed=0):
        init_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/bird_canny.png"
        )
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device="cpu").manual_seed(seed)

        return {
            "prompt": "corgi",
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            "output_type": "np",
            "generator": generator,
            "image": init_image,
        }

    def test_sd3_img2img_lora(self):
        pipe = self.pipeline_class.from_pretrained(self.repo_id, torch_dtype=torch.float16)
        pipe.load_lora_weights("nerijs/pixel-art-xl", weight_name="pixel-art-xl.safetensors")
        pipe.fuse_lora()
        pipe.unload_lora_weights()
        pipe = pipe.to("cuda")

        inputs = self.get_inputs(torch_device)

        image = pipe(**inputs).images[0]
        image_slice = image[0, :10, :10]
        print_tensor_test(image[0, -3:, -3:, -1].flatten())
        expected_slice = np.array(
            [
                0.47827148,
                0.5,
                0.71972656,
                0.3955078,
                0.4194336,
                0.69628906,
                0.37036133,
                0.40820312,
                0.6923828,
                0.36450195,
                0.40429688,
                0.6904297,
                0.35595703,
                0.39257812,
                0.68652344,
                0.35498047,
                0.3984375,
                0.68310547,
                0.34716797,
                0.3996582,
                0.6855469,
                0.3388672,
                0.3959961,
                0.6816406,
                0.34033203,
                0.40429688,
                0.6845703,
                0.34228516,
                0.4086914,
                0.6870117,
            ]
        )

        max_diff = numpy_cosine_similarity_distance(expected_slice.flatten(), image_slice.flatten())

        assert max_diff < 1e-4, f"Outputs are not close enough, got {max_diff}"
        pipe.unload_lora_weights()
        release_memory(pipe)
