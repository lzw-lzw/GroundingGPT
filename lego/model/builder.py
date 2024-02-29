#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import os
from transformers import AutoTokenizer
import torch
from lego.model import *
from lego import LEGOLlamaForCausalLM
from transformers import CLIPVisionModel, CLIPImageProcessor, AutoTokenizer
from video_llama.processors.video_processor import load_video
from video_llama.processors import AlproVideoTrainProcessor
from video_llama.models.ImageBind.data import load_and_transform_audio_data

DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IMAGE_START_TOKEN = "<im_start>"
DEFAULT_IMAGE_END_TOKEN = "<im_end>"
DEFAULT_VIDEO_PATCH_TOKEN = "<vi_patch>"
DEFAULT_VIDEO_START_TOKEN = "<vi_start>"
DEFAULT_VIDEO_END_TOKEN = "<vi_end>"
DEFAULT_SOUND_PATCH_TOKEN = "<so_patch>"
DEFAULT_SOUND_START_TOKEN = "<so_start>"
DEFAULT_SOUND_END_TOKEN = "<so_end>"

class Setting:
    def __init__(self):
        self.device = os.environ.get("LEGO_DEVICE", "cuda")
        # self.llasm_context_len = 2048
        self.sampling_rate = 16000
        self.image_token_len = 576
        self.video_token_len = 64
        self.sound_token_len = 8
        self.stop = "</s>"
CONFIG = Setting()

def load_pretrained_model(model_path, load_8bit=False, load_4bit=False, device_map="auto"):
    model = LEGOLlamaForCausalLM.from_pretrained(model_path,torch_dtype=torch.bfloat16).to(CONFIG.device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    tokenizer.add_tokens([DEFAULT_IMAGE_START_TOKEN, DEFAULT_IMAGE_END_TOKEN], special_tokens=True)
    tokenizer.add_tokens([DEFAULT_VIDEO_PATCH_TOKEN], special_tokens=True)
    tokenizer.add_tokens([DEFAULT_VIDEO_START_TOKEN, DEFAULT_VIDEO_END_TOKEN], special_tokens=True)
    tokenizer.add_tokens([DEFAULT_SOUND_PATCH_TOKEN], special_tokens=True)
    tokenizer.add_tokens([DEFAULT_SOUND_START_TOKEN, DEFAULT_SOUND_END_TOKEN], special_tokens=True)
    image_processor = CLIPImageProcessor.from_pretrained('openai/clip-vit-large-patch14-336' ,torch_dtype=torch.bfloat16)
    image_tower = CLIPVisionModel.from_pretrained('openai/clip-vit-large-patch14-336',torch_dtype=torch.bfloat16,low_cpu_mem_usage=True).to(CONFIG.device)
    image_config = image_tower.config
    image_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
    image_config.im_start_token, image_config.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_START_TOKEN, DEFAULT_IMAGE_END_TOKEN])
    image_config.use_im_start_end = True
    model.get_model().vision_tower[0] = image_tower
    
    video_transform = AlproVideoTrainProcessor(image_size=224, n_frms = model.config.max_frame).transform
    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048
    return model, tokenizer, image_processor, video_transform, context_len
