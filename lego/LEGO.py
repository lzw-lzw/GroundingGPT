from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import einops
from transformers import AutoConfig, AutoModelForCausalLM,LlamaConfig, LlamaModel, LlamaForCausalLM, CLIPVisionModel, CLIPImageProcessor
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from video_llama.models.Qformer import BertConfig, BertLMHeadModel
from video_llama.models.ImageBind.models.imagebind_model import ModalityType
from video_llama.models.ImageBind.models import imagebind_model
from video_llama.models.blip2 import Blip2Base, disabled_train
from lego.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IMAGE_START_TOKEN, DEFAULT_IMAGE_END_TOKEN, DEFAULT_VIDEO_PATCH_TOKEN, \
                           DEFAULT_VIDEO_START_TOKEN, DEFAULT_VIDEO_END_TOKEN, DEFAULT_SOUND_PATCH_TOKEN, DEFAULT_SOUND_START_TOKEN, DEFAULT_SOUND_END_TOKEN


class LEGOConfig(LlamaConfig):
    model_type = "LEGO"

class LEGOLlamaModel(LlamaModel):
    config_class = LEGOConfig

    def __init__(self, config: LlamaConfig):
        super(LEGOLlamaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = [CLIPVisionModel.from_pretrained(config.mm_vision_tower)]
            modules = [nn.Linear(config.mm_vision_hidden_size, config.hidden_size)]
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
            self.mm_vision_projector = nn.Sequential(*modules)

            self.visual_encoder, self.ln_vision = Blip2Base.init_vision_encoder(model_name="eva_clip_g",img_size=224, drop_path_rate=0, use_grad_checkpoint=False, precision='bf16')
            self.Qformer, self.query_tokens = Blip2Base.init_Qformer(config.num_query_token, self.visual_encoder.num_features)
            self.Qformer.cls = None
            self.Qformer.bert.embeddings.word_embeddings = None
            self.Qformer.bert.embeddings.position_embeddings = None
            for layer in self.Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
            self.llama_proj = nn.Linear(self.Qformer.config.hidden_size, config.hidden_size)
            self.video_frame_position_embedding = nn.Embedding(config.max_frame,self.Qformer.config.hidden_size)
            self.video_Qformer,self.video_query_tokens = self.init_video_Qformer(num_query_token = config.video_token_len, vision_width=768, num_hidden_layers =2)
            self.video_Qformer.cls = None
            self.video_Qformer.bert.embeddings.word_embeddings = None
            self.video_Qformer.bert.embeddings.position_embeddings = None
            for layer in self.video_Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None    

            self.audio_encoder,self.audio_hidden_size = imagebind_model.imagebind_huge()
            self.audio_encoder.load_state_dict(torch.load("{}/imagebind_huge.pth".format('/ckpt/imagebind')))

            self.audio_Qformer,self.audio_query_tokens = self.init_video_Qformer(num_query_token = config.sound_token_len, vision_width=1024, num_hidden_layers =2)
            self.audio_Qformer.cls = None
            self.audio_Qformer.bert.embeddings.word_embeddings = None
            self.audio_Qformer.bert.embeddings.position_embeddings = None
            for layer in self.audio_Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
            self.audio_llama_proj = nn.Linear(self.Qformer.config.hidden_size, config.hidden_size)
            self.audio_position_embedding = nn.Embedding(config.sound_token_len, 1024)

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_vision_mlp_adapter = model_args.pretrain_mm_vision_mlp_adapter

        self.config.mm_vision_tower = vision_tower
        self.config.image_token_len = model_args.image_token_len
        image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        if not hasattr(self, 'vision_tower'):
            vision_tower = CLIPVisionModel.from_pretrained(vision_tower)
        else:
            vision_tower = self.vision_tower[0]
        vision_tower.requires_grad_(False)
        vision_tower = vision_tower.to(torch.float16)
        self.vision_tower = [vision_tower]

        vision_config = vision_tower.config
        num_patches = (vision_config.image_size // vision_config.patch_size) ** 2

        self.config.use_mm_vision_proj = True
        self.config.mm_vision_hidden_size = vision_config.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature

        if not hasattr(self, 'mm_vision_projector'):
            modules = [nn.Linear(vision_config.hidden_size, self.config.hidden_size)]
            modules.append(nn.GELU())
            modules.append(nn.Linear(self.config.hidden_size, self.config.hidden_size))
            self.mm_vision_projector = nn.Sequential(*modules)

        if pretrain_mm_vision_mlp_adapter is not None:
            mm_vision_projector_weights = torch.load(pretrain_mm_vision_mlp_adapter, map_location='cpu')
            self.mm_vision_projector.load_state_dict({k.split('.')[-1]: v for k, v in mm_vision_projector_weights.items()})

        return dict(
            image_processor=image_processor,
            image_token_len=num_patches,
            vision_config=vision_config
        )


    @classmethod
    def init_video_Qformer(cls, num_query_token, vision_width,num_hidden_layers =2):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.num_hidden_layers = num_hidden_layers
        encoder_config.encoder_width = vision_width
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 1
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(torch.zeros(1, num_query_token, encoder_config.hidden_size))
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens
   

    def initialize_video_sound_modules(
        self, 
        model_args=None, 
        freeze_vit=True, 
        freeze_qformer=True,
        llama_proj_model=None,
        frozen_llama_proj=False,
        frozen_video_Qformer=False,
        frozen_audio_Qformer=False,
        ):
        self.config.num_query_token = model_args.num_query_token
        self.config.max_frame = model_args.max_frame
        self.config.video_token_len = model_args.video_token_len
        self.config.sound_token_len = model_args.sound_token_len
        if not hasattr(self, 'ln_vision'):
            self.visual_encoder, self.ln_vision = Blip2Base.init_vision_encoder(model_name="eva_clip_g",img_size=224, drop_path_rate=0, use_grad_checkpoint=False, precision='bf16')
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False               
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train

            for name, param in self.ln_vision.named_parameters():
                param.requires_grad = False               
            self.ln_vision = self.ln_vision.eval()
            self.ln_vision.train = disabled_train

            print("Loading VIT and freeze vision encoder and ln_vision")

        if not hasattr(self, 'Qformer'):      
            self.Qformer, self.query_tokens = Blip2Base.init_Qformer(model_args.num_query_token, self.visual_encoder.num_features)
            self.Qformer.cls = None
            self.Qformer.bert.embeddings.word_embeddings = None
            self.Qformer.bert.embeddings.position_embeddings = None
            for layer in self.Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None

        pretrained_state_dict = torch.load(model_args.q_former_model)["model"]
        model_dict = self.state_dict()
        model_dict_update = {k:v for k,v in pretrained_state_dict.items() if k in model_dict.keys()}
        model_dict.update(model_dict_update)
        self.load_state_dict(model_dict)

        print('load q_former checkpoint')

        if freeze_qformer:
            for name, param in self.Qformer.named_parameters():
                param.requires_grad = False
            self.Qformer = self.Qformer.eval()
            self.Qformer.train = disabled_train
            self.query_tokens.requires_grad = False
            print("freeze Qformer")
        print('Loading Q-Former Done')


        if not hasattr(self, 'llama_proj'):
            self.llama_proj = nn.Linear(self.Qformer.config.hidden_size, self.config.hidden_size)
        
        if llama_proj_model:
            print("load llama proj weight: {}".format(llama_proj_model))
            llama_proj_weight = torch.load(llama_proj_model, map_location="cpu")
            msg = self.load_state_dict(llama_proj_weight['model'], strict=False)

        if frozen_llama_proj:
            #  todo frozen  llama_proj
            for name, param in self.llama_proj.named_parameters():
                param.requires_grad = False
            print('LLAMA proj is frozen')
        else:
            for name, param in self.llama_proj.named_parameters():
                param.requires_grad = True
            print('LLAMA proj is not frozen')
        print('Loading llama_proj Done')
        if not hasattr(self, 'video_Qformer'):
            self.video_frame_position_embedding = nn.Embedding(model_args.max_frame, self.Qformer.config.hidden_size)
            self.num_video_query_token = model_args.video_token_len
            self.video_Qformer,self.video_query_tokens = self.init_video_Qformer(num_query_token = self.num_video_query_token,\
                vision_width=self.Qformer.config.hidden_size, num_hidden_layers =2)

        self.video_Qformer.cls = None
        self.video_Qformer.bert.embeddings.word_embeddings = None
        self.video_Qformer.bert.embeddings.position_embeddings = None
        for layer in self.video_Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None    
        
        if frozen_video_Qformer:
            #  todo frozen  llama_proj
            for name, param in self.video_Qformer.named_parameters():
                param.requires_grad = False
            for name, param in self.video_frame_position_embedding.named_parameters():
                param.requires_grad = False
            self.video_query_tokens.requires_grad = False
            
            print('video_Qformer is frozen')
        else:
            for name, param in self.video_Qformer.named_parameters():
                param.requires_grad = True
            for name, param in self.video_frame_position_embedding.named_parameters():
                param.requires_grad = True
            self.video_query_tokens.requires_grad = True
            print('video_Qformer is not frozen')
    
    
        # sound tower init
        print (f'Initializing audio encoder from {model_args.imagebind_model} ...')
        self.audio_encoder,self.audio_hidden_size = \
            imagebind_model.imagebind_huge()
        self.audio_encoder.load_state_dict(torch.load("{}/imagebind_huge.pth".format(model_args.imagebind_model)))
        # free sound encoder
        for name, param in self.audio_encoder.named_parameters():
            param.requires_grad = False
        self.audio_encoder.eval()
        print ('audio encoder initialized.')
        
        self.num_audio_query_token = model_args.sound_token_len
        
        if not hasattr(self, 'audio_Qformer'):
            self.audio_Qformer,self.audio_query_tokens = self.init_video_Qformer(num_query_token = self.num_audio_query_token,\
                vision_width=self.audio_hidden_size, num_hidden_layers =2)
        self.audio_Qformer.cls = None
        self.audio_Qformer.bert.embeddings.word_embeddings = None
        self.audio_Qformer.bert.embeddings.position_embeddings = None
        for layer in self.audio_Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
            
        if not hasattr(self, 'audio_llama_proj'):
            self.audio_llama_proj = nn.Linear(
                self.audio_Qformer.config.hidden_size, self.config.hidden_size
            )
            self.audio_position_embedding = nn.Embedding(8, self.audio_hidden_size)

        if frozen_audio_Qformer:
            for name, param in self.audio_Qformer.named_parameters():
                param.requires_grad = False
            self.audio_query_tokens.requires_grad = False
            for name, param in self.audio_llama_proj.named_parameters():
                param.requires_grad = False
            for name, param in self.audio_position_embedding.named_parameters():
                param.requires_grad = False
            print('audio_Qformer and audio-LLAMA proj is frozen')
        else:
            for name, param in self.audio_Qformer.named_parameters():
                param.requires_grad = True
            self.audio_query_tokens.requires_grad = True
            for name, param in self.audio_llama_proj.named_parameters():
                param.requires_grad = True
            for name, param in self.audio_position_embedding.named_parameters():
                param.requires_grad = True
            print('audio_Qformer is not frozen')

    def encode_videoQformer_visual(self, image):
        device = image.device
        batch_size,_,time_length,_,_ = image.size()
        image = einops.rearrange(image, 'b c t h w -> (b t) c h w')
        self.ln_vision=self.ln_vision.to(torch.float)
        image_embeds = self.ln_vision(self.visual_encoder(image)).to(device)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        # add frame_pos embedding
        position_ids = torch.arange(time_length, dtype=torch.long, device=query_tokens.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        frame_position_embeddings = self.video_frame_position_embedding(position_ids)
        q_hidden_state = query_output.last_hidden_state

        frame_position_embeddings = frame_position_embeddings.unsqueeze(-2)
        frame_hidden_state = einops.rearrange(q_hidden_state, '(b t) q h -> b t q h',b=batch_size,t=time_length)
        frame_hidden_state = frame_position_embeddings + frame_hidden_state

        # frame attention
        frame_hidden_state =  einops.rearrange(frame_hidden_state, 'b t q h -> b (t q) h',b=batch_size,t=time_length)
        frame_atts = torch.ones(frame_hidden_state.size()[:-1], dtype=torch.long).to(device)
        video_query_tokens = self.video_query_tokens.expand(frame_hidden_state.shape[0], -1, -1)

        video_query_output = self.video_Qformer.bert(
            query_embeds=video_query_tokens,
            encoder_hidden_states=frame_hidden_state,
            encoder_attention_mask=frame_atts,
            return_dict=True,
            )
        video_hidden = video_query_output.last_hidden_state

        inputs_llama = self.llama_proj(video_hidden)
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image_embeds.device)
        return inputs_llama, atts_llama


    def encode_audioQformer(self, audio,modality_type=ModalityType.AUDIO):
        device = audio.device
        audio_feature, audio_imagebind_finalout = self.audio_encoder.get_audio_feature(audio,modality_type=modality_type)
        batch_size,time_length = audio.size()[:2]

        position_ids = torch.arange(time_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        audio_position_embeddings = self.audio_position_embedding(position_ids)
        audio_imagebind_finalout = audio_imagebind_finalout + audio_position_embeddings

        audio_query_tokens = self.audio_query_tokens.expand(audio_imagebind_finalout.shape[0], -1, -1)
        frame_atts = torch.ones(audio_imagebind_finalout.size()[:-1], dtype=torch.long).to(device)

        audio_query_output = self.audio_Qformer.bert(
            query_embeds=audio_query_tokens, 
            encoder_hidden_states=audio_imagebind_finalout, 
            encoder_attention_mask=frame_atts,
            return_dict=True,
            )
        audio_hidden = audio_query_output.last_hidden_state

        inputs_llama = self.audio_llama_proj(audio_hidden)
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(device)
    
        return inputs_llama, atts_llama

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            images: Optional[torch.FloatTensor] = None,
            videos: Optional[torch.FloatTensor] = None,
            sounds: Optional[torch.FloatTensor] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        orig_embeds_params = getattr(self, 'orig_embeds_params', None)
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        #Vision embedding
        vision_tower = getattr(self, 'vision_tower', None)
        if vision_tower is not None and (input_ids.shape[1] != 1 or self.training) and images is not None:
            vision_tower = vision_tower[0]  
            with torch.no_grad():
                if type(images) is list:
                    # variable length images
                    image_features = []
                    for image in images:
                        if image is None:
                            image_feature = torch.zeros(1, self.config.image_token_len, vision_tower.config.hidden_size, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
                        else:
                            image_forward_out = vision_tower(image.unsqueeze(0), output_hidden_states=True)
                            select_hidden_state_layer = getattr(self.config, "mm_vision_select_layer", -1)
                            select_hidden_state = image_forward_out.hidden_states[select_hidden_state_layer]
                            image_feature = select_hidden_state[:, 1:]
                        image_features.append(image_feature)
                else:
                    image_forward_outs = vision_tower(images, output_hidden_states=True)
                    select_hidden_state_layer = getattr(self.config, "mm_vision_select_layer", -1)
                    select_hidden_state = image_forward_outs.hidden_states[select_hidden_state_layer]
                    image_features = select_hidden_state[:, 1:]
            if type(images) is list:
                image_features = [self.mm_vision_projector(image_feature)[0] for image_feature in image_features]
            else:
                image_features = self.mm_vision_projector(image_features)
            dummy_image_features = torch.zeros(self.config.image_token_len, vision_tower.config.hidden_size, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            dummy_image_features = self.mm_vision_projector(dummy_image_features)

            new_input_embeds = []
            cur_image_idx = 0
            for cur_input_ids, cur_input_embeds in zip(input_ids, inputs_embeds):
                if (cur_input_ids == vision_tower.config.im_patch_token).sum() == 0:
                    cur_input_embeds = cur_input_embeds + (0. * dummy_image_features).sum()
                    new_input_embeds.append(cur_input_embeds)
                    cur_image_idx += 1
                    continue
                if vision_tower.config.use_im_start_end:
                    cur_image_features = image_features[cur_image_idx]
                    num_patches = cur_image_features.shape[0]
                    if (cur_input_ids == vision_tower.config.im_start_token).sum() != (
                            cur_input_ids == vision_tower.config.im_end_token).sum():
                        raise ValueError("The number of image start tokens and image end tokens should be the same.")
                    image_start_tokens = torch.where(cur_input_ids == vision_tower.config.im_start_token)[0]
                    for image_start_token_pos in image_start_tokens:
                        cur_image_features = image_features[cur_image_idx].to(device=cur_input_embeds.device)
                        num_patches = cur_image_features.shape[0]
                        if cur_input_ids[image_start_token_pos + num_patches + 1] != vision_tower.config.im_end_token:
                            raise ValueError("The image end token should follow the image start token.")
                        if orig_embeds_params is not None:
                            cur_new_input_embeds = torch.cat((cur_input_embeds[:image_start_token_pos].detach(),
                                                              cur_input_embeds[image_start_token_pos:image_start_token_pos + 1],
                                                              cur_image_features, cur_input_embeds[
                                                                                  image_start_token_pos + num_patches + 1:image_start_token_pos + num_patches + 2],
                                                              cur_input_embeds[image_start_token_pos + num_patches + 2:].detach()), dim=0)
                        else:
                            cur_new_input_embeds = torch.cat((cur_input_embeds[:image_start_token_pos + 1], cur_image_features,
                                                              cur_input_embeds[image_start_token_pos + num_patches + 1:]), dim=0)
                        cur_image_idx += 1
                    new_input_embeds.append(cur_new_input_embeds)
                else:
                    cur_image_features = image_features[cur_image_idx]
                    num_patches = cur_image_features.shape[0]
                    if (cur_input_ids == vision_tower.config.im_patch_token).sum() != num_patches:
                        raise ValueError("The number of image patch tokens should be the same as the number of image patches.")
                    masked_indices = torch.where(cur_input_ids == vision_tower.config.im_patch_token)[0]
                    mask_index_start = masked_indices[0]
                    if (masked_indices != torch.arange(mask_index_start, mask_index_start + num_patches, device=masked_indices.device,
                                                       dtype=masked_indices.dtype)).any():
                        raise ValueError("The image patch tokens should be consecutive.")
                    if orig_embeds_params is not None:
                        cur_new_input_embeds = torch.cat((cur_input_embeds[:mask_index_start].detach(), cur_image_features,
                                                          cur_input_embeds[mask_index_start + num_patches:].detach()), dim=0)
                    else:
                        cur_new_input_embeds = torch.cat(
                            (cur_input_embeds[:mask_index_start], cur_image_features, cur_input_embeds[mask_index_start + num_patches:]),
                            dim=0)
                    new_input_embeds.append(cur_new_input_embeds)
            inputs_embeds = torch.stack(new_input_embeds, dim=0)

        
        #Video embedding    
        if videos is not None:
            # with torch.no_grad():
            if type(videos) is list:
                video_embeds = []
                for video in videos:
                    if video is None:
                        dummy_video = torch.zeros(3, 64, 224, 224, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
                        video_embed, _ = self.encode_videoQformer_visual(dummy_video.unsqueeze(0))
                    else:
                        video_embed, _ = self.encode_videoQformer_visual(video.unsqueeze(0))
                    video_embeds.append(video_embed)
            else:
                video_embeds = self.encode_videoQformer_visual(videos)
            
            new_input_embeds = []
            for cur_input_ids, cur_input_embeds, cur_video_embeds in zip(input_ids, inputs_embeds, video_embeds):
                if (cur_input_ids == self.config.video_patch_token).sum() == 0:
                    cur_input_embeds = cur_input_embeds + (0. * cur_video_embeds).sum()
                    new_input_embeds.append(cur_input_embeds)
                    continue
                if (cur_input_ids == self.config.video_start_token).sum() != (cur_input_ids == self.config.video_end_token).sum():
                    raise ValueError("The number of video start tokens and video end tokens should be the same.")

                video_start_tokens = torch.where(cur_input_ids == self.config.video_start_token)[0]
                if len(video_start_tokens) != len(cur_video_embeds):
                    raise ValueError(f"The number of video start tokens ({len(video_start_tokens)}) and video embeds ({len(cur_video_embeds)}) should be the same.")
                for video_start_token_pos, cur_video_embed in zip(video_start_tokens, cur_video_embeds):
                    cur_video_embed = cur_video_embed.to(device=cur_input_embeds.device)
                    num_patches = cur_video_embed.shape[0]
                    if cur_input_ids[video_start_token_pos + num_patches + 1] != self.config.video_end_token:
                        raise ValueError("The video end token should follow the video start token.")
                    if orig_embeds_params is not None:
                        cur_new_input_embeds = torch.cat(
                            (cur_input_embeds[:video_start_token_pos].detach(),
                             cur_input_embeds[video_start_token_pos:video_start_token_pos+1],
                             cur_video_embed,
                             cur_input_embeds[video_start_token_pos + num_patches + 1:video_start_token_pos + num_patches + 2],
                             cur_input_embeds[video_start_token_pos + num_patches + 2:].detach()), dim=0)
                    else:
                        cur_new_input_embeds = torch.cat((
                            cur_input_embeds[:video_start_token_pos+1],
                            cur_video_embed,
                            cur_input_embeds[video_start_token_pos + num_patches + 1:]), dim=0)
                new_input_embeds.append(cur_new_input_embeds)
            inputs_embeds = torch.stack(new_input_embeds, dim=0)

        #Sound embedding    
        if sounds is not None:
            # with torch.no_grad():
            if type(sounds) is list:
                sound_embeds = []
                for sound in sounds:
                    if sound is None:
                        dummy_sound = torch.zeros(1, 3, 1, 128, 204, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
                        sound_embed , _ = self.encode_audioQformer(dummy_sound)
                    else:
                        #  input audio shape [b t c h w]
                        sound_embed,_ = self.encode_audioQformer(sound)
                    sound_embeds.append(sound_embed)
            else:
                sound_embeds,_= self.encode_audioQformer(sounds)
                sound_embeds = [sound_embeds]
            
            new_input_embeds = []
            for cur_input_ids, cur_input_embeds, cur_sound_embeds in zip(input_ids, inputs_embeds, sound_embeds):
                if (cur_input_ids == self.config.sound_patch_token).sum() == 0:
                    cur_input_embeds = cur_input_embeds + (0. *cur_sound_embeds).sum()
                    new_input_embeds.append(cur_input_embeds)
                    continue
                if (cur_input_ids == self.config.sound_start_token).sum() != (cur_input_ids == self.config.sound_end_token).sum():
                    raise ValueError("The number of sound start tokens and sound end tokens should be the same.")

                sound_start_tokens = torch.where(cur_input_ids == self.config.sound_start_token)[0]
                if len(sound_start_tokens) != len(cur_sound_embeds):
                    raise ValueError(f"The number of sound start tokens ({len(sound_start_tokens)}) and sound embeds ({len(cur_sound_embeds)}) should be the same.")
                for sound_start_token_pos, cur_sound_embed in zip(sound_start_tokens, cur_sound_embeds):
                    cur_sound_embed = cur_sound_embed.to(device=cur_input_embeds.device)
                    num_patches = cur_sound_embed.shape[0]
                    if cur_input_ids[sound_start_token_pos + num_patches + 1] != self.config.sound_end_token:
                        raise ValueError("The sound end token should follow the sound start token.")
                    if orig_embeds_params is not None:
                        cur_new_input_embeds = torch.cat(
                            (cur_input_embeds[:sound_start_token_pos].detach(),
                             cur_input_embeds[sound_start_token_pos:sound_start_token_pos+1],
                             cur_sound_embed,
                             cur_input_embeds[sound_start_token_pos + num_patches + 1:sound_start_token_pos + num_patches + 2],
                             cur_input_embeds[sound_start_token_pos + num_patches + 2:].detach()), dim=0)
                    else:
                        cur_new_input_embeds = torch.cat((
                            cur_input_embeds[:sound_start_token_pos+1],
                            cur_sound_embed,
                            cur_input_embeds[sound_start_token_pos + num_patches + 1:]), dim=0)
                new_input_embeds.append(cur_new_input_embeds)
            inputs_embeds = torch.stack(new_input_embeds, dim=0)    
        return super(LEGOLlamaModel, self).forward(
            input_ids=None, attention_mask=attention_mask, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )


class LEGOLlamaForCausalLM(LlamaForCausalLM):
    config_class = LEGOConfig

    def __init__(self, config: LEGOConfig):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LEGOLlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_model(self):
        return self.model

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        videos: Optional[torch.FloatTensor] = None,
        sounds: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            images=images,
            videos=videos,
            sounds=sounds,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            # if torch.isnan(loss):
            #     raise ValueError('loss is nan')

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
                "videos": kwargs.get("videos", None),
                "sounds": kwargs.get("sounds", None),
            }
        )
        return model_inputs

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        vision_config = self.model.vision_tower[0].config
        vision_config.use_im_start_end = model_args.mm_use_im_start_end
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))
            vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IMAGE_START_TOKEN, DEFAULT_IMAGE_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))
            vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_START_TOKEN, DEFAULT_IMAGE_END_TOKEN])

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.pretrain_mm_vision_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_vision_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(
                        f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
        

    def initialize_video_sound_tokenizer(self,model_args,tokenizer):
        num_new_tokens = tokenizer.add_tokens([DEFAULT_VIDEO_PATCH_TOKEN], special_tokens=True)
        num_new_tokens += tokenizer.add_tokens([DEFAULT_VIDEO_START_TOKEN, DEFAULT_VIDEO_END_TOKEN], special_tokens=True)
        self.resize_token_embeddings(len(tokenizer))

        num_new_tokens += tokenizer.add_tokens([DEFAULT_SOUND_PATCH_TOKEN], special_tokens=True)
        num_new_tokens += tokenizer.add_tokens([DEFAULT_SOUND_START_TOKEN, DEFAULT_SOUND_END_TOKEN], special_tokens=True)
        self.resize_token_embeddings(len(tokenizer))

        if num_new_tokens > 0:
            input_embeddings = self.get_input_embeddings().weight.data
            output_embeddings = self.get_output_embeddings().weight.data

            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True)
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True)

            input_embeddings[-num_new_tokens:] = input_embeddings_avg
            output_embeddings[-num_new_tokens:] = output_embeddings_avg

        if model_args.tune_mm_mlp_adapter:
            self.get_model().orig_embeds_params = [self.get_input_embeddings().weight.data.clone()]
            for p in self.get_input_embeddings().parameters():
                p.requires_grad = True
            for p in self.get_output_embeddings().parameters():
                p.requires_grad = False

        if model_args.pretrain_mm_video_mlp_adapter and num_new_tokens > 0:
            mm_projector_weights = torch.load(model_args.pretrain_mm_video_mlp_adapter, map_location='cpu')
            embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
            assert num_new_tokens == 3
            if input_embeddings.shape == embed_tokens_weight.shape:
                input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
            elif embed_tokens_weight.shape[0] == num_new_tokens:
                input_embeddings[-num_new_tokens:] = embed_tokens_weight
            else:
                raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")

        video_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_VIDEO_PATCH_TOKEN])[0]
        video_start_token, video_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_VIDEO_START_TOKEN, DEFAULT_VIDEO_END_TOKEN])
        sound_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_SOUND_PATCH_TOKEN])[0]
        sound_start_token, sound_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_SOUND_START_TOKEN, DEFAULT_SOUND_END_TOKEN])
        self.model.config.video_patch_token = video_patch_token
        self.model.config.video_start_token = video_start_token
        self.model.config.video_end_token = video_end_token
        self.model.config.sound_patch_token = sound_patch_token
        self.model.config.sound_start_token = sound_start_token
        self.model.config.sound_end_token = sound_end_token

AutoConfig.register("LEGO", LEGOConfig)
AutoModelForCausalLM.register(LEGOConfig, LEGOLlamaForCausalLM)