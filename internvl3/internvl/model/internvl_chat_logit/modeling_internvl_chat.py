# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
import warnings
from typing import Any, List, Optional, Tuple, Union

import torch.utils.checkpoint
import transformers
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import (AutoModel, GenerationConfig, LlamaForCausalLM,
                          LlamaTokenizer)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, logging

from .configuration_internvl_chat import InternVLChatConfig
from .conversation import get_conv_template
from .modeling_intern_vit import InternVisionModel, has_flash_attn
from .modeling_internlm2 import InternLM2ForCausalLM
from pytorchvideo.models.hub import slowfast_r50
from peft import LoraConfig, get_peft_model
logger = logging.get_logger(__name__)


def version_cmp(v1, v2, op='eq'):
    import operator

    from packaging import version
    op_func = getattr(operator, op)
    return op_func(version.parse(v1), version.parse(v2))
        
class MLP(nn.Module):
    def __init__(self, input_dim=4096):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 16)
        self.fc5 = nn.Linear(16, 1)

        # 初始化线性层权重在 [0, 1] 之间
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                print('m.weight1', m.weight)
                # with torch.no_grad():
                m.weight.data.uniform_(0.0, 1e-2)
                print('m.weight2', m.weight)

                m.bias.data.zero_()
                print('m.bias', m.bias)
                # if m.bias is not None:
                #     nn.init.uniform_(m.bias, a=0.0, b=1.0)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        return x


class InternVLChatModel(PreTrainedModel):
    config_class = InternVLChatConfig
    main_input_name = 'pixel_values'
    base_model_prefix = 'language_model'
    _supports_flash_attn_2 = True
    _no_split_modules = ['InternVisionModel', 'LlamaDecoderLayer', 'InternLM2DecoderLayer']

    def __init__(self, config: InternVLChatConfig, vision_model=None, language_model=None, use_flash_attn=True):
        super().__init__(config)

        assert version_cmp(transformers.__version__, '4.36.2', 'ge')
        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        self.select_layer = config.select_layer
        self.template = config.template
        self.num_image_token = int((image_size // patch_size) ** 2 * (config.downsample_ratio ** 2))
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version
        use_flash_attn = use_flash_attn if has_flash_attn else False
        config.vision_config.use_flash_attn = True if use_flash_attn else False
        config.llm_config.attn_implementation = 'flash_attention_2' if use_flash_attn else 'eager'
        self.llm_arch_name = config.llm_config.architectures[0]
        print('this model')
        logger.info(f'num_image_token: {self.num_image_token}')
        logger.info(f'ps_version: {self.ps_version}')
        if vision_model is not None:
            self.vision_model = vision_model
        else:
            self.vision_model = InternVisionModel(config.vision_config)
        if language_model is not None:
            self.language_model = language_model
        else:
            if config.llm_config.architectures[0] == 'LlamaForCausalLM':
                self.language_model = LlamaForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'InternLM2ForCausalLM':
                self.language_model = InternLM2ForCausalLM(config.llm_config)
            else:
                raise NotImplementedError(f'{config.llm_config.architectures[0]} is not implemented.')

        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.llm_config.hidden_size

        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        )
        
        self.img_context_token_id = None
        self.conv_template = get_conv_template(self.template)
        self.system_message = self.conv_template.system_message
        if config.use_backbone_lora:
            self.wrap_backbone_lora(r=config.use_backbone_lora, lora_alpha=2 * config.use_backbone_lora)

        if config.use_llm_lora:
            self.wrap_llm_lora(r=config.use_llm_lora, lora_alpha=2 * config.use_llm_lora)

    def wrap_backbone_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        lora_config = LoraConfig(
            r=r,
            target_modules=['attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2'],
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        self.vision_model = get_peft_model(self.vision_model, lora_config)
        self.vision_model.print_trainable_parameters()

    def wrap_llm_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        # Determine the target modules based on the architecture of the language model
        if self.llm_arch_name == 'InternLM2ForCausalLM':
            target_modules = ['attention.wqkv', 'attention.wo', 'feed_forward.w1', 'feed_forward.w2', 'feed_forward.w3']
        elif self.llm_arch_name == 'Phi3ForCausalLM':
            target_modules = ['mlp.down_proj', 'mlp.gate_up_proj', 'self_attn.o_proj', 'self_attn.qkv_proj']
        elif self.llm_arch_name in ['Qwen2ForCausalLM', 'LlamaForCausalLM']:
            target_modules = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
                              'mlp.gate_proj', 'mlp.down_proj', 'mlp.up_proj']
        else:
            raise NotImplemented
        lora_config = LoraConfig(
            r=r,
            target_modules=target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            task_type='CAUSAL_LM'
        )
        self.language_model = get_peft_model(self.language_model, lora_config)
        self.language_model.enable_input_require_grads()
        self.language_model.print_trainable_parameters()
    def forward(
            self,
            # mos: torch.FloatTensor,
            pixel_values: torch.FloatTensor,
            # pixel_values2: torch.FloatTensor,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            image_flags: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()

        # Reshape pixel_values for ViT if it's 5D [B_llm, N_patches_dataset, C, H, W]
        # If it's 4D [B_vit, C, H, W] (e.g. from batch_chat or single image inference), it's handled by extract_feature
        if pixel_values.ndim == 5:
            B_llm, N_patches_dataset, C_img, H_img, W_img = pixel_values.shape
            pixel_values_reshaped = pixel_values.view(B_llm * N_patches_dataset, C_img, H_img, W_img)
            # image_flags should be [B_llm, N_patches_dataset]
            if image_flags is None: # Should be provided by dataloader for 5D pixel_values
                 raise ValueError("image_flags must be provided when pixel_values is 5D.")
            image_flags_flat = image_flags.view(-1).bool() # [B_llm * N_patches_dataset]
        elif pixel_values.ndim == 4: # Standard 4D input
            pixel_values_reshaped = pixel_values
            # If image_flags is provided for 4D input, it might be per-batch item, e.g. [B_vit]
            # For simplicity, assume all are active if 4D. Or image_flags needs clear definition for 4D.
            # The original logic `vit_embeds = vit_embeds[image_flags == 1]` assumed image_flags aligned with vit_embeds.shape[0]
            # Let's assume if image_flags are passed with 4D pixel_values, they are already 1D and aligned.
            if image_flags is not None:
                 image_flags_flat = image_flags.view(-1).bool()
            else: # Assume all active if no flags for 4D
                 image_flags_flat = torch.ones(pixel_values_reshaped.shape[0], dtype=torch.bool, device=pixel_values.device)
        else:
            raise ValueError(f"pixel_values has unsupported dimension: {pixel_values.ndim}")

        vit_embeds_extracted = self.extract_feature(pixel_values_reshaped)
        # vit_embeds_extracted shape: [pixel_values_reshaped.shape[0], self.num_image_token, llm_hidden_size]

        # Select features from active patches based on image_flags_flat
        vit_embeds_active_patches = vit_embeds_extracted[image_flags_flat]
        
        # Reshape for LLM: [Num_Active_Patches_Total * self.num_image_token, llm_hidden_size]
        final_vit_tokens_for_llm = vit_embeds_active_patches.reshape(-1, self.language_model.config.hidden_size)

        B_llm_embed, N_llm_embed, C_llm_embed = input_embeds.shape 
        input_embeds_flat = input_embeds.reshape(B_llm_embed * N_llm_embed, C_llm_embed)
        input_ids_flat = input_ids.reshape(B_llm_embed * N_llm_embed)
        selected_context_tokens_mask = (input_ids_flat == self.img_context_token_id)

        try:
            if final_vit_tokens_for_llm.shape[0] != selected_context_tokens_mask.sum().item():
                raise ValueError(
                    f"Mismatch between number of vision tokens ({final_vit_tokens_for_llm.shape[0]}) "
                    f"and <IMG_CONTEXT> tokens ({selected_context_tokens_mask.sum().item()}). Check image_flags logic. "
                    f"final_vit_tokens_for_llm.shape: {final_vit_tokens_for_llm.shape}, "
                    f"pixel_values.shape: {pixel_values.shape}, "
                    f"selected_context_tokens_mask.sum(): {selected_context_tokens_mask.sum().item()}, "
                    f"image_flags_flat.sum(): {image_flags_flat.sum().item()}"
                )
            input_embeds_flat[selected_context_tokens_mask] = final_vit_tokens_for_llm.to(input_embeds_flat.dtype)
        except Exception as e:
            logger.error(f'Error during visual token replacement: {e}')
            raise e

        input_embeds = input_embeds_flat.reshape(B_llm_embed, N_llm_embed, C_llm_embed)
        
        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states, # Set to True if needed by other parts
            return_dict=return_dict,
        )
        logits = outputs.logits
        loss = None
        
        shift_logits_for_loss_and_probs = None
        shift_labels_flat = None
        
        raw_logits_for_token_290 = None
        raw_logits_for_token_309 = None
        softmax2way_probs_tokens_290_309 = None
        full_softmax_prob_token_290 = None
        full_softmax_prob_token_309 = None

        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = CrossEntropyLoss()
            shift_logits_for_loss_and_probs = shift_logits.view(-1, self.language_model.config.vocab_size) 
            shift_labels_flat = shift_labels.view(-1)
            
            shift_labels_flat = shift_labels_flat.to(shift_logits_for_loss_and_probs.device)
            loss = loss_fct(shift_logits_for_loss_and_probs, shift_labels_flat)

            vocab_size = self.language_model.config.vocab_size
            token_id_290 = 290
            token_id_309 = 309
            
            num_flat_positions = shift_logits_for_loss_and_probs.shape[0]

            if token_id_290 < vocab_size and token_id_309 < vocab_size:
                raw_logits_for_token_290 = shift_logits_for_loss_and_probs[:, token_id_290]
                raw_logits_for_token_309 = shift_logits_for_loss_and_probs[:, token_id_309]
                
                stacked_specific_logits = torch.stack([raw_logits_for_token_290, raw_logits_for_token_309], dim=-1)
                softmax2way_probs_tokens_290_309 = torch.softmax(stacked_specific_logits, dim=-1)

                full_vocab_softmax_probs = torch.softmax(shift_logits_for_loss_and_probs, dim=-1)
                full_softmax_prob_token_290 = full_vocab_softmax_probs[:, token_id_290]
                full_softmax_prob_token_309 = full_vocab_softmax_probs[:, token_id_309]
            else:
                logger.warning(
                    f"Token ID {token_id_290} or {token_id_309} is out of vocab size ({vocab_size}). "
                    "Cannot compute specific probabilities/logits. Returning NaNs."
                )
                nan_tensor_flat = torch.full((num_flat_positions,), float('nan'), device=shift_logits_for_loss_and_probs.device, dtype=shift_logits_for_loss_and_probs.dtype)
                raw_logits_for_token_290 = nan_tensor_flat.clone()
                raw_logits_for_token_309 = nan_tensor_flat.clone()
                softmax2way_probs_tokens_290_309 = torch.full((num_flat_positions, 2), float('nan'), device=shift_logits_for_loss_and_probs.device, dtype=shift_logits_for_loss_and_probs.dtype)
                full_softmax_prob_token_290 = nan_tensor_flat.clone()
                full_softmax_prob_token_309 = nan_tensor_flat.clone()

        if not return_dict:
            # Mimic CausalLMOutputWithPast structure if not return_dict
            # This part might need careful adjustment based on how it's used without return_dict
            output_items = (logits,) + outputs[1:] # outputs[0] is logits
            if loss is not None:
                output_items = (loss,) + output_items
            # How to add probs_tokens_290_309 here is non-standard for tuple output
            return output_items

        # Prepare dictionary for CausalLMOutputWithPast or custom dict
        return_payload = {
            'loss': loss,
            'logits': logits, # Return full logits as per CausalLMOutputWithPast
            'past_key_values': outputs.past_key_values,
            'hidden_states': outputs.hidden_states,
            'attentions': outputs.attentions,
            # Custom fields:
            'label': shift_labels_flat, # (flattened shift_labels)
            'raw_shift_logits': shift_logits_for_loss_and_probs, # (flattened shift_logits)
            'predicted_token_ids': torch.argmax(shift_logits_for_loss_and_probs, dim=1) if shift_logits_for_loss_and_probs is not None else None,
            'raw_logits_for_token_290': raw_logits_for_token_290,
            'raw_logits_for_token_309': raw_logits_for_token_309,
            'softmax2way_probs_tokens_290_309': softmax2way_probs_tokens_290_309,
            'full_softmax_prob_token_290': full_softmax_prob_token_290,
            'full_softmax_prob_token_309': full_softmax_prob_token_309,
        }
        # Filter out None values if labels were not provided
        return {k: v for k, v in return_payload.items() if v is not None}

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        if self.ps_version == 'v1':
            warnings.warn("In ps_version 'v1', the height and width have not been swapped back, "
                          'which results in a transposed image.')
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values):
        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True).last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True).hidden_states[self.select_layer]
            
        vit_embeds = vit_embeds[:, 1:, :]
        
        E_B_vit = vit_embeds.shape[0]
        # Ensure num_vit_spatial_patches is a perfect square if taking sqrt
        num_vit_spatial_patches = vit_embeds.shape[1]
        if int(num_vit_spatial_patches**0.5)**2 != num_vit_spatial_patches:
            raise ValueError(
                f"The number of ViT spatial patches ({num_vit_spatial_patches}) is not a perfect square. "
                "Cannot reshape into (h, w)."
            )
        h_vit_spatial = w_vit_spatial = int(num_vit_spatial_patches ** 0.5)
        
        # Reshape to [E_B, h_vit_spatial, w_vit_spatial, vit_hidden_size]
        vit_embeds_reshaped = vit_embeds.reshape(E_B_vit, h_vit_spatial, w_vit_spatial, -1) 
        
        vit_embeds_shuffled = self.pixel_shuffle(vit_embeds_reshaped, scale_factor=self.downsample_ratio)
        # vit_embeds_shuffled shape: [E_B, h_s, w_s, c_s] 
        # where h_s * w_s = self.num_image_token (approximately, due to downsample_ratio)
        # and c_s = vit_hidden_size / (downsample_ratio_effective_channel_divider)

        vit_embeds_projected = self.mlp1(vit_embeds_shuffled) 
        # mlp1 output shape: [E_B, h_s, w_s, llm_hidden_size]

        # Reshape to [E_B, self.num_image_token, llm_hidden_size]
        _E_B, _h_s, _w_s, _C_llm = vit_embeds_projected.shape
        
        # self.num_image_token should be _h_s * _w_s
        # (image_size // patch_size)**2 * (downsample_ratio**2)
        # Check consistency:
        if _h_s * _w_s != self.num_image_token:
             logger.warning(
                 f"Mismatch in expected num_image_token ({self.num_image_token}) and "
                 f"actual from reshaped projected ViT embeds ({_h_s * _w_s}). This might cause issues. "
                 f"h_s={_h_s}, w_s={_w_s}"
             )
             # If they must match, this could be an error or require padding/truncation.
             # For now, we proceed with the actual number of tokens from projection.

        final_vit_embeds = vit_embeds_projected.reshape(_E_B, _h_s * _w_s, _C_llm)
        
        return final_vit_embeds

    def batch_chat(self, tokenizer, pixel_values, questions, generation_config, num_patches_list=None,
                   history=None, return_history=False, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>',
                   IMG_CONTEXT_TOKEN='<IMG_CONTEXT>', verbose=False, image_counts=None):
        if history is not None or return_history:
            print('Now multi-turn chat is not supported in batch_chat.')
            raise NotImplementedError

        if image_counts is not None:
            num_patches_list = image_counts
            print('Warning: `image_counts` is deprecated. Please use `num_patches_list` instead.')

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        queries = []
        for idx, num_patches in enumerate(num_patches_list):
            question = questions[idx]
            if pixel_values is not None and '<image>' not in question:
                question = '<image>\n' + question
            template = get_conv_template(self.template)
            template.system_message = self.system_message
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
            queries.append(query)

        tokenizer.padding_side = 'left'
        model_inputs = tokenizer(queries, return_tensors='pt', padding=True)
        input_ids = model_inputs['input_ids'].to(self.device)
        attention_mask = model_inputs['attention_mask'].to(self.device)
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        responses = tokenizer.batch_decode(generation_output, skip_special_tokens=True)
        responses = [response.split(template.sep)[0].strip() for response in responses]
        return responses

    def chat(self, tokenizer, pixel_values, question, generation_config, history=None, return_history=False,
             num_patches_list=None, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
             verbose=False):

        if history is None and pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question

        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        template = get_conv_template(self.template)
        template.system_message = self.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)

        history = [] if history is None else history
        for (old_question, old_answer) in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)

        model_inputs = tokenizer(query, return_tensors='pt')
        input_ids = model_inputs['input_ids'].to(self.device)
        attention_mask = model_inputs['attention_mask'].to(self.device)
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        response = response.split(template.sep)[0].strip()
        history.append((question, response))
        if return_history:
            return response, history
        else:
            query_to_print = query.replace(IMG_CONTEXT_TOKEN, '')
            query_to_print = query_to_print.replace(f'{IMG_START_TOKEN}{IMG_END_TOKEN}', '<image>')
            if verbose:
                print(query_to_print, response)
            return response
    
    @torch.no_grad()
    def generate(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **generate_kwargs,
    ) -> torch.LongTensor:

        assert self.img_context_token_id is not None
        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                vit_embeds = self.extract_feature(pixel_values)
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.img_context_token_id)
            assert selected.sum() != 0
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)

        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=True,
            **generate_kwargs,
        )

        return outputs

