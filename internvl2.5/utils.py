import os
import torch
import random
import numpy as np
from transformers import set_seed, AutoTokenizer
import json
import deepspeed
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
import torch
from torch.utils.data import Subset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import shutil
from torch.utils.data.dataloader import default_collate
import DST as DST
from einops import rearrange

NUM_DEBUG_SAMPLE = 10
class DataCollatorPadToMaxLen:

    def __init__(self, max_token_len, pad_token_id):
        self.max_token_len = max_token_len
        self.pad_token_id = pad_token_id

    def __call__(self, data):
        batch = {}
        input_ids = pad_sequence([default_collate(f['input_ids']) for f in data], 
                                  padding_value=self.pad_token_id, 
                                  batch_first=True)
        
        labels = pad_sequence([default_collate(f['labels']) for f in data],
                                   padding_value=DST.DEFAULT_LABEL_PADDING_NUM,
                                   batch_first=True)
        attention_mask = pad_sequence([default_collate(f['attention_mask']) for f in data],
                                        padding_value=0,
                                        batch_first=True)
        if isinstance(data[0]['image'], list):
            # if it is motion token, must not be dynamic frames
            frames_count = len(data[0]['image'][0][0])
        else:
            frames_count = len([f['image'][0].shape[0] for f in data])

        if isinstance(data[0]['image'], list):

            image = torch.concat([f['image'][0][0] for f in data], dim=0) #.reshape((len(data),) + data[0]["image"][0][0].shape)
            image = rearrange(image, '(bs fn) seq hid -> bs fn seq hid', bs=len(data))
            # print(image.shape)
            motion = torch.concat([f['image'][0][1] for f in data], dim=0) #.reshape((len(data),) + data[0]["image"][0][1].shape)
            motion = rearrange(motion, '(bs fn) hid -> bs fn hid', bs=len(data))
            # print(motion.shape)
        else:
            _len = len(data[0]["image"][0].shape)
            image = torch.concat([default_collate(f['image']) for f in data], dim=0).reshape((-1,) + data[0]["image"][0].shape[-_len:])
        image_id = [f['image_id'] for f in data]
        image_num = [f['image_num'] for f in data]
        batch['input_ids'] = input_ids
        batch['labels'] = labels
        batch['attention_mask'] = attention_mask
        batch['image'] = image
        batch['image_num'] = image_num
        batch['frames_count'] = frames_count
        batch['image_id'] = image_id
        if isinstance(data[0]['image'], list):
            batch['motion'] = motion
        return batch

def print_rank_0(msg, rank=None):
    if rank is not None and rank <= 0:
        print(msg)
    elif is_rank_0():
        print(msg)

def is_rank_0():
    """Check whether it is rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            return True
        else:
            return False
    else:
        return True

def get_rank():
    """Check whether it is rank 0."""
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    else:
        return 0

def to_device(batch, device):
    output = {}
    for k, v in batch.items():
        try:
            output[k] = v.to(device)
        except:
            output[k] = v
    return output


class MovingAverage:

    def __init__(self):
        self.count = 0
        self.total = 0
        self.mean = 0

    def update(self, num):
        self.total += num
        self.count += 1
        self.mean = self.total / self.count

        return self.mean


def set_random_seed(seed):
    if seed is not None:
        set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_all_reduce_mean(tensor):
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    tensor = tensor / torch.distributed.get_world_size()
    return tensor


def get_optimizer_grouped_parameters(model,
                                     weight_decay,
                                     no_decay_name_list=[
                                         "bias", "LayerNorm.weight"
                                     ],
                                     small_learning_rate_list=
                                     ["embed"], small_lr=1e-4):
    
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if (not any(nd in n
                            for nd in no_decay_name_list) and (not any(nd in n
                            for nd in small_learning_rate_list)) and p.requires_grad)
            ],
            "weight_decay":
            weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (any(nd in n
                        for nd in no_decay_name_list) and (not any(nd in n
                            for nd in small_learning_rate_list)) and p.requires_grad)
            ],
            "weight_decay":
            0.0,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (not any(nd in n
                            for nd in no_decay_name_list) and (any(nd in n
                            for nd in small_learning_rate_list)) and p.requires_grad)
            ],
            "weight_decay":
            weight_decay,
            "lr": small_lr
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (any(nd in n
                        for nd in no_decay_name_list) and (any(nd in n
                            for nd in small_learning_rate_list)) and p.requires_grad)
            ],
            "weight_decay":
            0.0,
            "lr": small_lr
        },
    ]
    return optimizer_grouped_parameters


def _z3_params_to_fetch(param_list):
    return [
        p for p in param_list
        if hasattr(p, 'ds_id') and p.ds_status == ZeroParamStatus.NOT_AVAILABLE
    ]


def moving_average(model, model_ema, beta=0.992, device=None, zero_stage=0):
    zero_stage_3 = (zero_stage == 3)
    with torch.no_grad():
        for param, param_ema in zip(model.parameters(),
                                    model_ema.parameters()):
            # TODO: use prefiltering for efficiency
            params_to_fetch = _z3_params_to_fetch([param, param_ema
                                                   ]) if zero_stage_3 else []
            should_gather_param = len(params_to_fetch) > 0
            with deepspeed.zero.GatheredParameters(
                    params_to_fetch, enabled=should_gather_param):
                data = param.data
                if device is not None:
                    data = data.to(device)
                param_ema.data.copy_(torch.lerp(data, param_ema.data, beta))

def save_hf_format(model, tokenizer, args, sub_folder=""):
    # used to save huggingface format, so we can use it for hf.from_pretrained
    model_to_save = model.module if hasattr(model, 'module') else model
    CONFIG_NAME = "config.json"
    WEIGHTS_NAME = "pytorch_model.bin"
    output_dir = os.path.join(args.output_dir, sub_folder)
    os.makedirs(output_dir, exist_ok=True)
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)
    save_dict = model_to_save.state_dict()
    # for key in list(save_dict.keys()):
    #     if "lora" in key:
    #         del save_dict[key]
    torch.save(save_dict, output_model_file)
    try:
        model_to_save.config.to_json_file(output_config_file)
    except:
        args_dict = vars(args)
        torch.save(args_dict,os.path.join(output_dir, 'train_args.pt'))
        print ("config can't be saved")
    # tokenizer.save_vocabulary(output_dir)
    tokenizer.save_pretrained(output_dir)  # this will save all tokenizer files

def save_zero_three_model(model_ema, global_rank, save_dir, zero_stage=0, sub_folder=""):
    zero_stage_3 = (zero_stage == 3)
    output_dir = os.path.join(save_dir, sub_folder)
    os.makedirs(output_dir, exist_ok=True)
    WEIGHTS_NAME = "pytorch_model.bin"
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)

    model_to_save = model_ema.module if hasattr(model_ema,
                                                'module') else model_ema
    if not zero_stage_3:
        if global_rank == 0:
            torch.save(model_to_save.state_dict(), output_model_file)
    else:
        output_state_dict = {}
        for k, v in model_to_save.named_parameters():

            if hasattr(v, 'ds_id'):
                with deepspeed.zero.GatheredParameters(_z3_params_to_fetch([v
                                                                            ]),
                                                       enabled=zero_stage_3):
                    v_p = v.data.clone().detach().cpu() # this is a hack to get around the fact that we can't get the data from the param
            else:
                v_p = v.cpu()
            if global_rank == 0 and "lora" not in k:
                output_state_dict[k] = v_p
        if global_rank == 0:
            torch.save(output_state_dict, output_model_file)
        del output_state_dict
