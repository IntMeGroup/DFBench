# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, List, Literal, Optional

import torch
import torch.nn.functional as F

from swift.utils import get_env_args, is_deepspeed_enabled
from swift.llm.template.base import Template
from swift.llm.template.constant import LLMTemplateType, MLLMTemplateType
from swift.llm.template.register import register_template
from swift.llm.template.template_inputs import StdTemplateInputs
from swift.llm.template.template_meta import TemplateMeta
from swift.llm.template.utils import Context, Word, findall
from swift.llm.template.vision_utils import load_audio, load_batch
from swift.llm.template.template.llama import Llama3TemplateMeta
from swift.llm.template.template.utils import DEFAULT_SYSTEM, ChatmlTemplateMeta


class Qwen2VLTemplate_Customized(Template):
    image_token_id = 151655
    video_token_id = 151656
    version = 'v2_5'

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        from qwen_vl_utils import fetch_image, fetch_video
        assert media_type in {'image', 'video'}
        if media_type == 'image':
            inputs.images[index] = fetch_image({'image': inputs.images[index]})
            return ['<|vision_start|><|image_pad|><|vision_end|>']
        else:
            inputs.videos[index] = fetch_video({'video': inputs.videos[index]}).to(torch.uint8)
            return ['<|vision_start|><|video_pad|><|vision_end|>']

    def replace_ref(self, ref: str, index: int, inputs: StdTemplateInputs) -> List[Context]:
        return [f'<|object_ref_start|>{ref}<|object_ref_end|>']

    def replace_bbox(self, bbox: List[int], index: int, inputs: StdTemplateInputs) -> List[Context]:
        return [f'<|box_start|>{self._get_bbox_str(bbox)}<|box_end|>']

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        processor = self.processor
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        images = inputs.images
        videos = inputs.videos
        for media_type in ['images', 'videos']:
            if locals()[media_type]:
                if media_type == 'images':
                    media_token = self.image_token_id
                    media_inputs = processor.image_processor(
                        images=images, videos=None, return_tensors='pt', do_resize=False)
                    media_grid_thw = media_inputs['image_grid_thw']
                else:
                    media_inputs = processor.image_processor(
                        images=None, videos=videos, return_tensors='pt', do_resize=False)
                    media_grid_thw = media_inputs['video_grid_thw']
                    media_token = self.video_token_id
                    if self.version == 'v2_5':
                        from qwen_vl_utils import vision_process
                        media_inputs['second_per_grid_ts'] = [
                            processor.image_processor.temporal_patch_size / vision_process.FPS
                        ] * len(media_grid_thw)
                idx_list = findall(input_ids, media_token)
                added_tokens_len = 0
                for i, idx in enumerate(idx_list):
                    merge_length = processor.image_processor.merge_size**2
                    token_len = (media_grid_thw[i].prod() // merge_length)
                    input_ids = input_ids[:idx
                                          + added_tokens_len] + [media_token] * token_len + input_ids[added_tokens_len
                                                                                                      + idx + 1:]
                    if labels:
                        labels = labels[:idx + added_tokens_len] + [-100] * token_len + labels[added_tokens_len + idx
                                                                                               + 1:]
                    added_tokens_len += token_len - 1
                encoded.update(media_inputs)

        encoded['input_ids'] = input_ids
        encoded['labels'] = labels
        return encoded

    def _post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if not self.is_training:
            return inputs
        input_ids = inputs['input_ids']
        _model = model.model
        if not hasattr(_model, 'embed_tokens'):
            _model = _model.model  # LoRA
        pixel_values = inputs.get('pixel_values')
        pixel_values_videos = inputs.get('pixel_values_videos')
        image_grid_thw = inputs.get('image_grid_thw')
        video_grid_thw = inputs.get('video_grid_thw')
        second_per_grid_ts = inputs.get('second_per_grid_ts')
        mos = inputs.get('mos')

        inputs_embeds = _model.embed_tokens(input_ids)

        dtype = model.visual.get_dtype() if self.version == 'v2' else model.visual.dtype
        if pixel_values is None and pixel_values_videos is None:  # plain-text
            if is_deepspeed_enabled():
                from PIL import Image
                images = [Image.new('RGB', (32, 32), (0, 0, 0))]
                media_inputs = self.processor.image_processor(images=images, videos=None, return_tensors='pt')
                device = input_ids.device
                pixel_values = media_inputs['pixel_values'].to(device)

                pixel_values = pixel_values.type(dtype)
                image_embeds = model.visual(pixel_values, grid_thw=media_inputs['image_grid_thw'])
                inputs_embeds += image_embeds.mean() * 0.
        else:
            if pixel_values is not None:
                pixel_values = pixel_values.type(dtype)
                image_embeds = model.visual(pixel_values, grid_thw=image_grid_thw)
                image_mask = (input_ids == model.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(dtype)
                video_embeds = model.visual(pixel_values_videos, grid_thw=video_grid_thw)
                video_mask = (input_ids == model.config.video_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        # fix https://github.com/huggingface/transformers/pull/33487
        kwargs = {}
        if self.version == 'v2_5':
            kwargs = {'second_per_grid_ts': second_per_grid_ts}
        position_ids, _ = model.get_rope_index(
            input_ids, image_grid_thw, video_grid_thw, attention_mask=inputs['attention_mask'], **kwargs)
        return {'inputs_embeds': inputs_embeds, 'position_ids': position_ids.contiguous(), 'mos': mos}

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super()._data_collator(batch, padding_to=padding_to)
        second_per_grid_ts = self.gather_list(batch, 'second_per_grid_ts')
        if second_per_grid_ts:
            res['second_per_grid_ts'] = second_per_grid_ts
        for media_type in ['image', 'video']:
            grid_thw = [b[f'{media_type}_grid_thw'] for b in batch if b.get(f'{media_type}_grid_thw') is not None]
            if grid_thw:
                res[f'{media_type}_grid_thw'] = torch.concat(grid_thw)
        return res
