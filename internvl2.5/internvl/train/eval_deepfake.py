import gc
import json
import logging
import math
import os
import random
import sys
import traceback
import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import torch
import torch.distributed as dist
import transformers
from internvl.dist_utils import init_dist
from internvl.model.internlm2.modeling_internlm2 import InternLM2ForCausalLM
from internvl.model.internvl_chat22 import (InternVisionConfig,
                                          InternVisionModel,
                                          InternVLChatConfig,
                                          InternVLChatModel)
from internvl.patch import (concat_pad_data_collator,
                            replace_llama_rmsnorm_with_fused_rmsnorm,
                            replace_train_sampler)
from internvl.train.constants import (BOX_END_TOKEN, BOX_START_TOKEN,
                                      IMG_CONTEXT_TOKEN, IMG_END_TOKEN,
                                      IMG_START_TOKEN, QUAD_END_TOKEN,
                                      QUAD_START_TOKEN, REF_END_TOKEN,
                                      REF_START_TOKEN)
from internvl.train.dataset import (ConcatDataset, TCSLoader,
                                    WeightedConcatDataset, build_transform,
                                    dynamic_preprocess, preprocess,
                                    preprocess_internlm, preprocess_mpt,
                                    preprocess_phi3)
from internvl.train.trainer_monkey_patch import replace_create_optimizer
from PIL import Image, ImageFile, PngImagePlugin, UnidentifiedImageError
from torch.utils.data import Dataset
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          HfArgumentParser, Trainer, TrainingArguments,
                          set_seed)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.logging import (enable_default_handler,
                                        enable_explicit_format, set_verbosity)
import pandas as pd
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr, kendalltau
# Apply necessary patches for the transformers library
replace_llama_rmsnorm_with_fused_rmsnorm()
replace_train_sampler()

# Try to import petrel_client for image loading, fallback to PIL if unavailable
try:
    from petrel_client.client import Client
    from petrel_client.common.config import Config
    has_tcs_loader = True
except ImportError as E:
    print('petrel_client is not installed. Using PIL to load images.')
    has_tcs_loader = False

# Set constants for image processing and logging
IGNORE_INDEX = -100
IGNORE_INDEX2 = 92542
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

os.environ['TOKENIZERS_PARALLELISM'] = 'true'
@dataclass
class CustomArguments:
    """
    Custom arguments for additional configurations like output files.
    """
    output_file: Optional[str] = field(
        default='results.csv',
        metadata={'help': 'The name of the CSV file to save evaluation results.'}
    )
    metrics_file: Optional[str] = field(
        default='metrics.txt',
        metadata={'help': 'The name of the TXT file to save evaluation metrics.'}
    )
    
@dataclass
class ModelArguments:
    """
    Arguments for specifying model, tokenizer, and configurations.
    """
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to pretrained model or model identifier from huggingface.co/models'}
    )
    vision_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to pretrained model or model identifier from huggingface.co/models'}
    )
    llm_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to pretrained model or model identifier from huggingface.co/models'}
    )
    mlp_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to pretrained model or model identifier from huggingface.co/models'}
    )
    freeze_llm: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the LLM decoder.'},
    )
    freeze_backbone: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the vision backbone of the model.'},
    )
    freeze_mlp: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the MLP layers of the model.'},
    )
    unfreeze_vit_layers: int = field(
        default=0,
        metadata={'help': 'Specify the number of ViT layers to unfreeze. Default is 0.'},
    )
    vision_select_layer: int = field(
        default=-1,
        metadata={'help': 'Specify the layer of ViT feature map to use. Default is last layer.'},
    )
    use_backbone_lora: int = field(
        default=0,
        metadata={'help': 'Set the LoRA adapter rank for the backbone model. Default is 0.'}
    )
    use_llm_lora: int = field(
        default=0,
        metadata={'help': 'Set the LoRA adapter rank for the LLM. Default is 0.'}
    )
    unfreeze_lm_head: bool = field(
        default=False,
        metadata={'help': "Set to True to unfreeze the language model's head."},
    )
    use_custom_trainer: bool = field(
        default=False,
        metadata={'help': 'Set to True to enable the use of a custom trainer.'},
    )
    grad_checkpoint: Optional[bool] = field(
        default=False,
        metadata={'help': 'Set to True to use gradient checkpointing.'},
    )
    drop_path_rate: float = field(
        default=0.0,
        metadata={'help': 'Set the drop path rate for the ViT model. Default is 0.'},
    )
    ps_version: str = field(
        default='v2',
        metadata={'help': 'Specify the version of pixel shuffle implementation. Default is `v1`.'
                          'Please use `v2` to fix the bug of transposed image.'}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments for specifying data input for training and evaluation.
    """
    max_seq_length: Optional[int] = field(
        default=2048,
        metadata={
            'help': (
                'The maximum total input sequence length after tokenization. Sequences longer '
                'than this will be truncated, sequences shorter will be padded.'
            )
        },
    )
    force_image_size: Optional[int] = field(
        default=448,
        metadata={'help': 'Set the desired size for the image. Default is 224.'},
    )
    down_sample_ratio: Optional[float] = field(
        default=0.5,
        metadata={'help': 'Set the desired down-sampling ratio for the image. Default is 1.0.'},
    )
    pad2square: Optional[bool] = field(
        default=False,
        metadata={'help': 'Pad the image to a square shape if set to True.'},
    )
    conv_style: Optional[str] = field(
        default='internlm2-chat', metadata={'help': 'Prompt style for a conversation.'}
    )
    meta_path: Optional[str] = field(
        default=None,
        metadata={'help': 'The path of the meta file of datasets.'},
    )
    use_data_resampling: Optional[bool] = field(
        default=False,
        metadata={'help': 'Set to True to use data resampling.'},
    )
    dynamic_image_size: Optional[bool] = field(
        default=False,
        metadata={'help': 'Set to True to use dynamic image size.'},
    )
    use_thumbnail: Optional[bool] = field(
        default=False,
        metadata={'help': 'Set to True to add a thumbnail image.'},
    )
    min_dynamic_patch: Optional[int] = field(
        default=1,
        metadata={'help': 'The minimum number of dynamic patches. Default is 1.'},
    )
    max_dynamic_patch: Optional[int] = field(
        default=12,
        metadata={'help': 'The maximum number of dynamic patches. Default is 6.'},
    )
    normalize_type: Optional[str] = field(
        default='imagenet',
        metadata={'help': 'The normalize type for the image. Default is imagenet.'},
    )

       
class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        template_name,
        meta,
        train_eval,
        tokenizer,
        tcs_loader,
        ds_name,
        num_image_token,
        image_size=224,
        is_train=True,
        pad2square=False,
        group_by_length=False,
        dynamic_image_size=False,
        use_thumbnail=False,
        min_dynamic_patch=1,
        max_dynamic_patch=6,
        min_num_frame=4,  # for video data
        max_num_frame=12,  # for video data
        sampling_method='rand',  # for video data
        repeat_time=1,
        normalize_type='imagenet',
        random_seed=0,
    ):
        super(LazySupervisedDataset, self).__init__()
        self.ds_name = ds_name
        self.tokenizer = tokenizer
        self.template_name = template_name
        self.num_image_token = num_image_token
        logger.info(f'[Dataset] num_image_token: {num_image_token}')
        logger.info(f'[Dataset] dynamic_image_size: {dynamic_image_size}')
        logger.info(f'[Dataset] use_thumbnail: {use_thumbnail}')
        logger.info(f'[Dataset] min_dynamic_patch: {min_dynamic_patch}, max_dynamic_patch: {max_dynamic_patch}')

        self.image_size = image_size
        self.is_train = is_train
        self.pad2square = pad2square
        self.max_num_frame = max_num_frame
        self.min_num_frame = min_num_frame
        self.sampling_method = sampling_method

        logger.info('Formatting inputs...Skip in lazy mode')
        #assert meta['annotation'].endswith('jsonl'), f'annotation must be jsonl, but got {meta["annotation"]}'

        total_ranks = torch.distributed.get_world_size()
        current_rank = torch.distributed.get_rank()

        """
        This section of the code is used to read hundreds of millions of data entries.
        By using caching and splitting the data according to rank, it ensures fast reading
        speed and prevents out-of-memory.
        """
        if train_eval=='eval':
            with open(meta['annotation_test'], 'r') as f:
                self.raw_data = f.readlines()
                if repeat_time < 1:
                    # If repeat_time is less than 1, select a portion of the data
                    self.raw_data = self.raw_data[:int(len(self.raw_data) * repeat_time)]
                if repeat_time > 1:
                    assert isinstance(repeat_time, int)
                    # Repeat the list if repeat_time is greater than 1
                    self.raw_data = self.raw_data * repeat_time
        

        # self.rng = np.random.default_rng(seed=random_seed)
        # # if self.force_shuffle:
        # self.rng.shuffle(self.raw_data)

        self.root = meta['root']
        self.cached_data_dict = {}
        self.tcs_loader = tcs_loader
        self.group_by_length = group_by_length
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.normalize_type = normalize_type

        # If the precomputed length does not exist, roughly estimate the length of
        # each sample to improve the efficiency of group_by_length.
        if self.group_by_length:
            self.conv2length = {}  # Using a dictionary to speed up token length calculation
            self.length = []
            for data_item in self.raw_data:
                data_item = json.loads(data_item)
                if 'length' in data_item:
                    token_length = data_item['length']  # Use precomputed length if available
                else:
                    # Compute token length using the tokenizer
                    conversations = '\n'.join([temp['value'] for temp in data_item['conversations']])
                    str_length = len(conversations)
                    if str_length not in self.conv2length:
                        token_length = tokenizer(
                            conversations, return_tensors='pt', padding=False, truncation=False,
                        ).input_ids.size(1)
                        self.conv2length[str_length] = token_length + num_image_token * (
                                    max_dynamic_patch + use_thumbnail)
                    else:
                        token_length = self.conv2length[str_length]
                self.length.append(token_length)

    def __len__(self):
        return len(self.raw_data)

    def get_preprocess_function(self):
        # Select the appropriate preprocessing function based on the template name
        if self.template_name == 'Hermes-2':
            preprocess_function = preprocess_mpt
        elif self.template_name == 'internlm2-chat':
            preprocess_function = preprocess_internlm
        elif self.template_name == 'phi3-chat':
            preprocess_function = preprocess_phi3
        elif self.template_name == 'internvl2_5':
            preprocess_function = preprocess_internvl2_5
        else:
            preprocess_function = preprocess
        return preprocess_function

    def load_image(self, image_path):
        # Load the image using tcs_loader if available, otherwise use PIL
        if self.tcs_loader is not None and 's3://' in image_path:
            return self.tcs_loader(image_path)
        image = Image.open(image_path).convert('RGB')
        image_resized = image.resize((1024,1024))
        return image_resized

    def get_image_path(self, image_path):
        if image_path.startswith('s3://'):  # for ceph
            image_path = self.root + image_path
        else:  # for local image
            image_path = os.path.join(self.root, image_path)
        return image_path

    def get_transform(self):
        # Build transformation function
        transform = build_transform(is_train=self.is_train, input_size=self.image_size,
                                    pad2square=self.pad2square, normalize_type=self.normalize_type)
        return transform

    ##my
    def multi_modal_get_item(self, data_item):
        # Build transformation function
        transform = self.get_transform()

        # Ensure the first conversation contains an image placeholder
        if '<image>' not in data_item['conversations'][0]['value']:
            data_item['conversations'][0]['value'] = '<image>\n' + data_item['conversations'][0]['value']

        # Merge the image path
        image_path = self.get_image_path(data_item['image'])

        # Load the image using tcs_loader if available, otherwise use PIL
        image = self.load_image(image_path)

        if self.dynamic_image_size:  # If dynamic image size is enabled, preprocess the image dynamically
            images = dynamic_preprocess(image, min_num=self.min_dynamic_patch, max_num=self.max_dynamic_patch,
                                        image_size=self.image_size, use_thumbnail=self.use_thumbnail)
        else:  # Otherwise, use the original image as a single patch
            images = [image]

        # Apply the transformation to each image and stack the results into a tensor
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)

        # Ensure that there is only one patch if dynamic image size is not enabled
        num_patches = pixel_values.size(0)
        if not self.dynamic_image_size:
            assert num_patches == 1, f'The number of patches should be 1, but got {num_patches}.'

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()

        # Preprocess the conversations and generate the return dictionary
        ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                  self.tokenizer, [self.num_image_token * num_patches],
                                  group_by_length=self.group_by_length, ds_name=self.ds_name)

        # Create the final return dictionary
        # mos = torch.tensor(float(data_item['mos']), dtype=torch.bfloat16)
        answer = data_item['conversations'][1]['value']
        # Create the final return dictionary
        ret = dict(
            imgname = data_item['image'],
            answer = answer,
            # mos = mos,
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long)
        )
        return ret
    
    def multi_modal_multi_image_get_item(self, data_item):
        # Build transformation function
        transform = self.get_transform()

        images, num_tiles = [], []
        num_image = len(data_item['image'])
        for image_path in data_item['image']:
            # Merge the image path
            image_path = self.get_image_path(image_path)
            # Load the image using tcs_loader if available, otherwise use PIL
            image = self.load_image(image_path)
            if self.dynamic_image_size:  # If dynamic image size is enabled, preprocess the image dynamically
                image = dynamic_preprocess(image, min_num=self.min_dynamic_patch,
                                           max_num=max(1, self.max_dynamic_patch // num_image),
                                           image_size=self.image_size, use_thumbnail=self.use_thumbnail)
                images += image
                num_tiles.append(len(image))
            else:  # Otherwise, use the original image as a single patch
                images.append(image)
                num_tiles.append(1)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()

        # Preprocess the conversations and generate the return dictionary
        num_image_tokens = [self.num_image_token * num_tile for num_tile in num_tiles]
        ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                  self.tokenizer, num_image_tokens, group_by_length=self.group_by_length,
                                  use_packed_ds=self.use_packed_ds, ds_name=self.ds_name, num_image=num_image)

        # Calculate position_ids for packed dataset
        position_ids = ret['attention_mask'].long().cumsum(-1) - 1
        position_ids.masked_fill_(ret['attention_mask'] == 0, 1)
        image_end_token_id = self.tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
        assert (ret['input_ids'][0] == image_end_token_id).sum() == num_image, f'image tokens are truncated, this dataset is {self.ds_name}'

        # Create the final return dictionary
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            position_ids=position_ids[0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long)
        )
        return ret

    def video_get_item(self, data_item):
        # Build transformation function
        transform = self.get_transform()

        # Ensure the first conversation contains a video placeholder
        if '<video>' not in data_item['conversations'][0]['value']:
            data_item['conversations'][0]['value'] = '<video>\n' + data_item['conversations'][0]['value']

        # Get the video file path
        video_file = data_item['video']
        video_path = os.path.join(self.root, video_file)

        # Load the video frames using tcs_loader
        # TODO: Load videos without using tcsloader.
        image_list = self.tcs_loader(
            video_path,
            image_type='video',
            max_num_frames=self.max_num_frame,
            min_num_frames=self.min_num_frame,
            sample=self.sampling_method,
            clip=data_item.get('clip', None))

        # Generate special tokens for each video frame
        special_tokens = '\n'.join(['Frame-{}: <image>'.format(i + 1) for i in range(len(image_list))])
        data_item['conversations'][0]['value'] = data_item['conversations'][0]['value'].replace(
            '<video>\n', special_tokens + '\n')

        # Transform each frame image and stack them into a tensor
        pixel_values = [transform(image) for image in image_list]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()

        # Preprocess the conversations and generate the return dictionary
        num_image_tokens = [self.num_image_token] * num_patches
        ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                  self.tokenizer, num_image_tokens, group_by_length=self.group_by_length,
                                  use_packed_ds=self.use_packed_ds, ds_name=self.ds_name, num_image=num_patches)

        # Calculate position_ids for packed dataset
        position_ids = ret['attention_mask'].long().cumsum(-1) - 1
        position_ids.masked_fill_(ret['attention_mask'] == 0, 1)

        # Create the final return dictionary
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            position_ids=position_ids[0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long)
        )
        return ret

    def pure_text_get_item(self, data_item):
        # Build transformation function
        transform = self.get_transform()

        # Create a blank white image
        image = Image.new('RGB', (224, 224), (255, 255, 255))

        # Dynamically preprocess the image to generate patches
        images = dynamic_preprocess(image, min_num=self.min_dynamic_patch, max_num=1,
                                    image_size=self.image_size, use_thumbnail=self.use_thumbnail)

        # Apply the transformation to each image patch and stack them into a tensor
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)

        # Ensure there is only one patch
        assert num_patches == 1, f'The number of patches should be 1, but got {num_patches}.'

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()

        # Preprocess the conversations and generate the return dictionary
        ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                  self.tokenizer, [self.num_image_token * num_patches], text_only=True,
                                  group_by_length=self.group_by_length, use_packed_ds=self.use_packed_ds,
                                  ds_name=self.ds_name)

        # Calculate position_ids for packed dataset
        position_ids = ret['attention_mask'].long().cumsum(-1) - 1
        position_ids.masked_fill_(ret['attention_mask'] == 0, 1)

        # Create the final return dictionary
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            position_ids=position_ids[0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([0] * num_patches, dtype=torch.long)
        )
        return ret

    def _enable_worker_distributed(self):
        if (
            self.distributed_mode
            and not self.worker_distributed
            and self.worker_id is not None
        ):
            self.worker_distributed = True
            self.raw_data = self.raw_data[self.worker_id::self.num_workers]
            logger.info(f'worker_distributed is enabled, {self.num_workers=}, {len(self.raw_data)=}')

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i >= len(self.raw_data):
            if self.use_packed_ds:
                raise NotImplementedError
            else:
                i = i % len(self.raw_data)

        try_cnt, max_try = 0, 10
        while True:
            if try_cnt > max_try:
                raise StopIteration
            try:
                data_item = json.loads(self.raw_data[i])
                # conversations = data_item['conversations']
                # check_conversations_repetition(conversations, repeat_threshold=0.4, ngram=10)
                if 'image' in data_item and len(data_item['image']) != 0:
                    if type(data_item['image']) == list:
                        ret = self.multi_modal_multi_image_get_item(data_item)
                    else:
                        ret = self.multi_modal_get_item(data_item)
                elif 'video' in data_item and data_item['video'] is not None and data_item['video'] != '':
                    ret = self.video_get_item(data_item)
                else:
                    ret = self.pure_text_get_item(data_item)
                break
            except Exception as e:
                try_cnt += 1
                print(e, self.ds_name, flush=True)
                if not isinstance(e, (UnidentifiedImageError, FileNotFoundError)):
                    traceback.print_exc()
                data_item = json.loads(self.raw_data[i])
                if 'image' in data_item:
                    if type(data_item['image']) == list:
                        images = [self.root + item for item in data_item['image']]
                        print(f'Failed to load image: {images}, the dataset is: {self.ds_name}')
                    else:
                        if data_item['image'].startswith('s3://'):
                            data_path = self.root + data_item['image']
                        else:
                            data_path = os.path.join(self.root, data_item['image'])
                        print(f'Failed to load image: {data_path}, the dataset is: {self.ds_name}')
                elif 'video' in data_item:
                    data_path = os.path.join(self.root, data_item['video'])
                    print(f'Failed to load video: {data_path}, the dataset is: {self.ds_name}')
                i = random.randint(0, len(self.raw_data) - 1)
        return ret

    def __iter__(self):
        self._enable_worker_distributed()
        start_idx = 0

        assert self.worker_state_key is not None
        if self.worker_state_key in self._state_dict and len(self._state_dict[self.worker_state_key]) > 0:
            start_idx = self._state_dict[self.worker_state_key]['current_idx']

            self._state_dict.pop(self.worker_state_key)

        if self.worker_id == 0:
            logger.info(
                f'[{self.ds_name}] [Worker id {self.worker_id}] '
                f'begin to iter with {start_idx=}'
            )

        for i in range(start_idx, len(self)):
            yield self[i]



def build_datasets(
    data_args,
    train_eval,
    tokenizer,
    tcs_loader,
    model,
    group_by_length=False,
    dynamic_image_size=False,
    use_thumbnail=False,
    min_dynamic_patch=1,
    max_dynamic_patch=12,
    normalize_type='imagenet',
):
    datasets = []
    lengths = []
    ds_collections = json.loads(open(data_args.meta_path).read())
    for ds_idx, ds_name in enumerate(ds_collections.keys()):
        repeat_time = ds_collections[ds_name]['repeat_time']
        if 'max_dynamic_patch' in ds_collections[ds_name]:
            max_num = ds_collections[ds_name]['max_dynamic_patch']
            logger.info(f'max_dynamic_patch is set to {max_num} according to the meta file')
        else:
            max_num = max_dynamic_patch
        dataset = LazySupervisedDataset(
            data_args.conv_style, ds_collections[ds_name],train_eval,
            tokenizer,
            tcs_loader,
            ds_name=ds_name,
            num_image_token=model.num_image_token,
            image_size=data_args.force_image_size,
            is_train=ds_collections[ds_name]['data_augment'],
            pad2square=data_args.pad2square,
            group_by_length=group_by_length,
            dynamic_image_size=dynamic_image_size,
            use_thumbnail=use_thumbnail,
            min_dynamic_patch=min_dynamic_patch,
            max_dynamic_patch=max_num,
            repeat_time=repeat_time,
            normalize_type=normalize_type,
            random_seed=ds_idx,
        )
        logger.info(f'Add dataset: {ds_name} with length: {len(dataset)}')
        datasets.append(dataset)
        if data_args.use_data_resampling:
            lengths.append(math.sqrt(len(dataset)))
        else:
            lengths.append(len(dataset))
    if data_args.use_data_resampling:
        total_length = sum(lengths)
        weights = [l / total_length for l in lengths]
        train_dataset = WeightedConcatDataset(datasets, weights)
    else:
        train_dataset = ConcatDataset(datasets)
    return train_dataset

def load_lora_weights(model, lora_weight_path):
    lora_state_dict = torch.load(lora_weight_path)
    model.load_state_dict(lora_state_dict, strict=False)
    logger.info(f'Loaded LoRA weights from {lora_weight_path}')
import pandas as pd
from scipy.stats import spearmanr, pearsonr, kendalltau
def save_and_evaluate(data_list, output_file='results3.csv', metrics_file='metrics3.txt'):
    # data_list items are expected to be: [imgname, ground_truth_answer, model_output, prob_A, prob_B]
    total = len(data_list)
    right = 0
    for item_row in data_list:
        if len(item_row) < 3: # Ensure there are enough elements
            print(f"Skipping malformed row for accuracy calculation: {item_row}")
            continue
        ground_truth = str(item_row[1]).strip().upper() # answer (e.g., "A" or "B")
        model_pred = str(item_row[2]).strip().upper()   # output (e.g., "A" or "B")
        
        if ground_truth == model_pred:
            right += 1
            
    accuracy = right / total if total > 0 else 0
    
    # Convert the list into a pandas DataFrame
    df = pd.DataFrame(data_list, columns=['imgname', 'answer', 'output', 'prob_A', 'prob_B'])

    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f'Results saved to {output_file}')
    print(f"Accuracy: {accuracy:.4f}")
    # Save metrics to a .txt file
    with open(metrics_file, 'a') as f: # Append mode
        f.write(f"Dataset/Run: {output_file}\n")
        f.write(f"Total items: {total}\n")
        f.write(f"Correct predictions: {right}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
    
def main():
    # Parse input arguments
    # See all possible arguments in src/transformers/training_args.py
    # If use DeepSpeed zero3, init_dist must before HfArgumentParser
    launcher = os.environ.get('LAUNCHER', 'slurm')
    init_dist(launcher=launcher, backend='nccl')
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, CustomArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith('.json'):
        # If we pass only one argument to the script, and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, custom_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, custom_args = parser.parse_args_into_dataclasses()
    training_args.evaluation_strategy = "steps"
    training_args.do_eval =True
    training_args.per_device_eval_batch_size =1
    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    # send_example_telemetry('InternV-Chat', model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    set_verbosity(log_level)
    enable_default_handler()
    enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f'Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}'
        + f'distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}'
    )
    logger.info(f'Training/evaluation parameters {training_args}')

    # Detecting last checkpoint and eventually continue from last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f'Output directory ({training_args.output_dir}) already exists and is not empty. '
                'Use --overwrite_output_dir to overcome.'
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f'Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change '
                'the `--output_dir` or add `--overwrite_output_dir` to train from scratch.'
            )
    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load pretrained model, tokenizer, and image processor
    tokenizer_path = model_args.model_name_or_path or model_args.llm_path
    logger.info(f'Loading Tokenizer: {tokenizer_path}')
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, add_eos_token=False, trust_remote_code=True, use_fast=False)
    tokenizer.tokenizer_path = tokenizer_path
    tokenizer.model_max_length = data_args.max_seq_length
    token_list = [IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN,
                  QUAD_START_TOKEN, QUAD_END_TOKEN, REF_START_TOKEN,
                  REF_END_TOKEN, BOX_START_TOKEN, BOX_END_TOKEN]
    num_new_tokens = tokenizer.add_tokens(token_list, special_tokens=True)
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    tcs_loader = TCSLoader('~/petreloss.conf') if has_tcs_loader else None

    if model_args.model_name_or_path is not None:
        logger.info('Loading InternVLChatModel...')
        config = InternVLChatConfig.from_pretrained(model_args.model_name_or_path)
        config.vision_config.drop_path_rate = model_args.drop_path_rate
        if config.llm_config.model_type == 'internlm2':
            config.llm_config.attn_implementation = 'flash_attention_2'  # for InternLM
            logger.info('Using flash_attention_2 for InternLM')
        else:
            config.llm_config._attn_implementation = 'flash_attention_2'  # for LLaMA
            logger.info('Using flash_attention_2 for LLaMA')
        config.template = data_args.conv_style
        config.select_layer = model_args.vision_select_layer
        config.dynamic_image_size = data_args.dynamic_image_size
        config.use_thumbnail = data_args.use_thumbnail
        config.ps_version = model_args.ps_version
        config.min_dynamic_patch = data_args.min_dynamic_patch
        config.max_dynamic_patch = data_args.max_dynamic_patch
        model = InternVLChatModel.from_pretrained(
            model_args.model_name_or_path, torch_dtype=torch.bfloat16, config=config)
        try:
            lora_weight_path = os.path.join(model_args.model_name_or_path,"lora_weights.pth")
            load_lora_weights(model, lora_weight_path)  # Add this line to load LoRA weights
        except:
            pass
    else:
        logger.info('Loading ViT-6B...')
        vision_config = InternVisionConfig.from_pretrained(model_args.vision_path)
        vision_config.drop_path_rate = model_args.drop_path_rate
        vision_model = InternVisionModel.from_pretrained(
            model_args.vision_path, torch_dtype=torch.bfloat16, config=vision_config)
        logger.info('Loading LLaMA...')
        llm_config = AutoConfig.from_pretrained(model_args.llm_path, trust_remote_code=True)
        if llm_config.model_type == 'internlm2':
            model_type = InternLM2ForCausalLM
            llm_config.attn_implementation = 'flash_attention_2'  # for InternLM
            logger.info('Using flash_attention_2 for InternLM')
        else:
            model_type = AutoModelForCausalLM
            llm_config._attn_implementation = 'flash_attention_2'  # for LLaMA
            logger.info('Using flash_attention_2 for LLaMA')
        llm = model_type.from_pretrained(
            model_args.llm_path, torch_dtype=torch.bfloat16,
            config=llm_config, trust_remote_code=True)
        logger.info('Building InternVLChatConfig...')
        internvl_chat_config = InternVLChatConfig(
            vision_config.to_dict(), llm_config.to_dict(), downsample_ratio=data_args.down_sample_ratio,
            pad2square=data_args.pad2square, template=data_args.conv_style,
            select_layer=model_args.vision_select_layer, dynamic_image_size=data_args.dynamic_image_size,
            use_thumbnail=data_args.use_thumbnail, ps_version=model_args.ps_version,
            min_dynamic_patch=data_args.min_dynamic_patch, max_dynamic_patch=data_args.max_dynamic_patch)
        internvl_chat_config.force_image_size = data_args.force_image_size
        logger.info('Building InternVLChatModel...')
        model = InternVLChatModel(internvl_chat_config, vision_model, llm)
    model.img_context_token_id = img_context_token_id

    assert model.config.downsample_ratio == data_args.down_sample_ratio

    if model_args.mlp_path is not None:
        logger.info('Loading pretrained MLP projector...')
        state_dict = torch.load(model_args.mlp_path, map_location='cpu')
        message = model.mlp1.load_state_dict(state_dict)
        logger.info(message)
    logger.info('Finished')

    patch_size = model.config.vision_config.patch_size
    logger.info(f'model.config.force_image_size: {model.config.force_image_size}')
    logger.info(f'data_args.force_image_size: {data_args.force_image_size}')
    logger.info(f'model.config.vision_config.image_size: {model.config.vision_config.image_size}')
    if model.config.vision_config.image_size != data_args.force_image_size:
        logger.info(f'Resizing position embedding from '
                    f'{model.config.vision_config.image_size} '
                    f'to {data_args.force_image_size}...')
        model.vision_model.resize_pos_embeddings(old_size=model.config.vision_config.image_size,
                                                 new_size=data_args.force_image_size,
                                                 patch_size=patch_size)
        model.config.vision_config.image_size = data_args.force_image_size
    model.config.force_image_size = data_args.force_image_size
    model.num_image_token = int((data_args.force_image_size // patch_size) ** 2 * (data_args.down_sample_ratio ** 2))

    if num_new_tokens > 0:
        model.language_model.resize_token_embeddings(len(tokenizer))
        output_embeddings = model.language_model.get_output_embeddings().weight.data
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

        model.config.llm_config.vocab_size = len(tokenizer)
        model.language_model.config.vocab_size = len(tokenizer)

    model.language_model.config.use_cache = False
    model.vision_model.gradient_checkpointing = True
    model.vision_model.encoder.gradient_checkpointing = True
    if model_args.grad_checkpoint:
        model.language_model._set_gradient_checkpointing()
    eval_dataset = build_datasets(
        data_args, 'eval', tokenizer, tcs_loader, model, group_by_length=training_args.group_by_length,
        dynamic_image_size=data_args.dynamic_image_size, use_thumbnail=data_args.use_thumbnail,
        min_dynamic_patch=data_args.min_dynamic_patch, max_dynamic_patch=data_args.max_dynamic_patch,
        normalize_type=data_args.normalize_type)
    print('eval_dataset',eval_dataset)
    def _freeze_params(module):
        for param in module.parameters():
            param.requires_grad = False

    if model_args.freeze_backbone:
        # model.vision_model = model.vision_model.eval()
        _freeze_params(model.vision_model)

    if model_args.freeze_llm:
        model.language_model = model.language_model.eval()
        _freeze_params(model.language_model)

    if model_args.unfreeze_lm_head:
        model.language_model.lm_head.requires_grad = True

    if model_args.use_backbone_lora:
        model.wrap_backbone_lora(r=model_args.use_backbone_lora, lora_alpha=2 * model_args.use_backbone_lora)
        model.config.use_backbone_lora = model_args.use_backbone_lora

    if model_args.use_llm_lora:
        model.wrap_llm_lora(r=model_args.use_llm_lora, lora_alpha=2 * model_args.use_llm_lora)
        model.config.use_llm_lora = model_args.use_llm_lora

    if model_args.freeze_mlp:
        _freeze_params(model.mlp1)

    if model_args.unfreeze_vit_layers != 0:
        layers = model.vision_model.encoder.layers[model_args.unfreeze_vit_layers:]
        for k, v in layers.named_parameters():
            logger.info(f'Unfreezing ViT layer: {k}')
            v.requires_grad = True

    # print trainable parameters
    if dist.get_rank() == 0:
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info(name)

    # set seed for torch dataloaders
    set_seed(training_args.seed)

    # Initialize our Trainer
    if model_args.use_custom_trainer:
        replace_create_optimizer()
    print('training_args',training_args)
    
    set_seed(training_args.seed)

    # Initialize our Trainer
    from torch.utils.data import DataLoader
    if model_args.use_custom_trainer:
        replace_create_optimizer()
    model.eval()
    model.cuda()
   
    # Get token IDs for A and B
    # Ensure these are single tokens as expected by the prompt "Just answer with A or B"
    token_A_id = tokenizer.encode('A', add_special_tokens=False)
    token_B_id = tokenizer.encode('B', add_special_tokens=False)

    if len(token_A_id) != 1:
        print(f"Warning: Token 'A' is not a single token: {token_A_id}. Using first token if available, or UNK.")
        token_A_id_val = tokenizer.unk_token_id if not token_A_id else token_A_id[0]
    else:
        token_A_id_val = token_A_id[0]

    if len(token_B_id) != 1:
        print(f"Warning: Token 'B' is not a single token: {token_B_id}. Using first token if available, or UNK.")
        token_B_id_val = tokenizer.unk_token_id if not token_B_id else token_B_id[0]
    else:
        token_B_id_val = token_B_id[0]
    
    print(f"Using Token ID for 'A': {token_A_id_val}")
    print(f"Using Token ID for 'B': {token_B_id_val}")
    if token_A_id_val == tokenizer.unk_token_id or token_B_id_val == tokenizer.unk_token_id:
        print("CRITICAL WARNING: 'A' or 'B' maps to UNK token. Probabilities will be unreliable.")

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=1  # Ensuring batch_size is 1 for simplicity in indexing
    )


    data_list = []
    with torch.no_grad():
        for item2 in tqdm(eval_dataloader, desc="Evaluating"):

            item = {}
            for k, v in item2.items():
                if k!='imgname' and k!='answer': # and k!='mos' (mos is now part of item)
                    item[k] = v.to(model.device)
            
            imgname = item2['imgname'][0]
            answer = item2['answer'][0] # This is the ground truth string, e.g., "A" or "B"
            
            output =  model(
                # mos=item["mos"].to(torch.bfloat16), # Assuming mos is batched, take first
                pixel_values=item["pixel_values"].to(torch.bfloat16), # Assuming pixel_values is already batched correctly
                input_ids=item["input_ids"],
                attention_mask=item["attention_mask"],
                image_flags=item["image_flags"], # Assuming image_flags is batched
                labels=item["labels"]
            )
            
            # Extract predictions and logits
            # output['label'] is shift_labels (flattened)
            # output['raw_shift_logits'] is shift_logits (flattened)
            # output['predicted_token_ids'] is argmax(shift_logits) (flattened)
            
            shift_labels = output['label'] # Flattened shift_labels
            raw_shift_logits = output['raw_shift_logits'] # Flattened raw_shift_logits
            predicted_token_ids_flat = output['predicted_token_ids']

            # Find the part of the sequence corresponding to the answer for decoding
            # This logic attempts to mirror the original way decoded_output was found
            active_label_indices = torch.where((shift_labels != IGNORE_INDEX) & (shift_labels != IGNORE_INDEX2))[0]
            
            decoded_output = ""
            prob_A, prob_B = -1.0, -1.0  # Default values

            if len(active_label_indices) > 0:
                # The original code for decoded_output was:
                # filtered_list = [x for x in output['label'] if x != IGNORE_INDEX and x != IGNORE_INDEX2] # output['label'] was shift_labels
                # decoded_output = tokenizer.decode(output['logit'][-len(filtered_list) - 1:-1]) # output['logit'] was argmax
                # This implies the answer is at a specific position relative to the end of active labels.
                # For simplicity, let's decode the predicted tokens at active_label_indices
                
                # Let's use the first active label index as the position for A/B probability
                answer_token_position_in_flat_logits = active_label_indices[0]
                
                # Decode the model's prediction for the answer part
                # This is simplified; the original logic was more complex.
                # We assume the relevant prediction is at `answer_token_position_in_flat_logits`
                predicted_answer_tokens = predicted_token_ids_flat[answer_token_position_in_flat_logits : answer_token_position_in_flat_logits + 1] # Assuming single token A/B
                decoded_output = tokenizer.decode(predicted_answer_tokens).strip()

                # Get the logits at this specific position
                logits_at_answer_pos = raw_shift_logits[answer_token_position_in_flat_logits] # Shape: (vocab_size)

                if token_A_id_val != tokenizer.unk_token_id and token_B_id_val != tokenizer.unk_token_id:
                    logit_A_tensor = logits_at_answer_pos[token_A_id_val]
                    logit_B_tensor = logits_at_answer_pos[token_B_id_val]
                    
                    # Apply softmax
                    softmax_input = torch.tensor([logit_A_tensor, logit_B_tensor], device=logit_A_tensor.device, dtype=torch.float32)
                    softmax_probs = torch.softmax(softmax_input, dim=0)
                    prob_A = softmax_probs[0].item()
                    prob_B = softmax_probs[1].item()
                else:
                    print(f"Warning: Token ID for 'A' or 'B' is UNK. Cannot compute probabilities for {imgname}.")
            else:
                print(f"Warning: No active labels found for {imgname}. Cannot determine answer position.")
                decoded_output = "[NO_ACTIVE_LABELS]"

            with open('data_list3.txt', 'a') as f:
                f.write(f"{imgname};{answer};{decoded_output};{prob_A};{prob_B}\n")
            data_list.append([imgname, answer, decoded_output, prob_A, prob_B])

    # Save results to CSV and calculate SRCC, PLCC, KRCC
    output_file=custom_args.output_file
    metrics_file=custom_args.metrics_file
    save_and_evaluate(data_list, output_file=output_file, metrics_file=metrics_file)

if __name__ == '__main__':
    main()   

