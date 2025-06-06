   # Copyright (c) Alibaba, Inc. and its affiliates.
# Part of the implementation is borrowed from huggingface/transformers.
import os
from contextlib import contextmanager, nullcontext
from functools import wraps
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from peft import PeftModel
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import Seq2SeqTrainer as HfSeq2SeqTrainer
from transformers import Trainer as HfTrainer
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.utils import is_peft_available

from swift.utils import JsonlWriter, Serializer
from swift.trainers.arguments import Seq2SeqTrainingArguments, TrainingArguments
from swift.trainers.mixin import SwiftMixin
from swift.trainers.torchacc_mixin import TorchAccMixin

from tqdm import tqdm


class Seq2SeqTrainer(TorchAccMixin, SwiftMixin, HfSeq2SeqTrainer):
    args: Seq2SeqTrainingArguments

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = True  # fix transformers>=4.46.2
        if self.args.predict_with_generate:
            from swift.llm import PtEngine
            self.infer_engine = PtEngine.from_model_template(
                self.model, self.template, max_batch_size=self.args.per_device_eval_batch_size)
        self.jsonl_writer = JsonlWriter(os.path.join(self.args.output_dir, 'predict.jsonl'))
        self.eval_loss_list = []

    @staticmethod
    def _predict_data_collator(batch):
        return {'_data': batch}

    @contextmanager
    def _patch_predict_with_generate(self):
        origin_mode = self.template.mode
        self.template.set_mode('pt')
        is_multimodal = self.model.model_meta.is_multimodal
        origin_data_collator = self.data_collator

        if is_multimodal:
            self.template.remove_post_encode_hook()
        self.data_collator = self._predict_data_collator
        try:
            yield
        finally:
            if is_multimodal:
                self.template.register_post_encode_hook([self.model])
            self.data_collator = origin_data_collator
            self.template.set_mode(origin_mode)


    def evaluate(self, *args, **kwargs):
        print('--------------- Evaluate! ---------------')
        self.model.eval()
        eval_dataset = self.eval_dataset
        from torch.utils.data import DataLoader
        if eval_dataset is None:
            raise ValueError("Evaluation dataset is None. Please provide a valid evaluation dataset.")
        
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=1,
            collate_fn=self.data_collator,
        )

        mos_values = []
        predicted_values = []
        # Loop through the evaluation dataset
        with torch.no_grad():
            for item in tqdm(eval_dataloader):
                output = self.model(**item)
                score1 = output['score1'].item()

                mos = float(item["mos"][0].item())
                mos_values.append(mos)
                predicted_values.append(score1)
                self.eval_loss_list.append(float(mos) - float(score1))


        from scipy.stats import spearmanr, pearsonr, kendalltau
        srcc = spearmanr(mos_values, predicted_values)[0]
        plcc = pearsonr(mos_values, predicted_values)[0]
        krcc = kendalltau(mos_values, predicted_values)[0]
    
        print(f'SRCC_score: {srcc}')
        print(f'PLCC_score: {plcc}')
        print(f'KRCC_score: {krcc}')

        return {"eval_loss": srcc}

    def save_eval_loss_image(self, save_path):
        import matplotlib.pyplot as plt

        # Assuming you store SRCC scores over multiple evaluations
        evaluation_steps = list(range(1, len(self.eval_loss_list) + 1))  # Generate step numbers
        evaluation_steps = [x for x in evaluation_steps]

        # Plot SRCC scores
        plt.figure()
        plt.plot(evaluation_steps, self.eval_loss_list, linestyle='-', color='orange')
        plt.xlabel("Evaluation Step")
        plt.ylabel("L1 loss")
        plt.title("eval/loss")
        plt.grid(True)

        # Save the figure
        os.makedirs(save_path, exist_ok=True)
        plot_path = os.path.join(save_path, "eval_loss.png")
        plt.savefig(plot_path)


    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
        **gen_kwargs,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys)
        from swift.llm import RequestConfig, InferRequest
        data_list = inputs['_data']
        labels_list = [InferRequest.remove_response(data['messages']) for data in data_list]
        resp_list = self.infer_engine.infer(
            data_list,
            RequestConfig(max_tokens=self.model.generation_config.max_new_tokens),
            use_tqdm=False,
            template=self.template)

        response_list = []
        device = self.args.device
        for data, resp, labels in zip(data_list, resp_list, labels_list):
            response = resp.choices[0].message.content
            self.jsonl_writer.append({'response': response, 'labels': labels, **data})
            response_list.append(Serializer.to_tensor(resp.choices[0].message.content).to(device=device))
        labels_list = [Serializer.to_tensor(labels).to(device=device) for labels in labels_list]
        response_list = pad_sequence(response_list, batch_first=True, padding_value=0)
        labels_list = pad_sequence(labels_list, batch_first=True, padding_value=0)
        return None, response_list, labels_list

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss_kwargs = {}
        labels = None
        if (self.label_smoother is not None or self.compute_loss_func is not None) and 'labels' in inputs:
            labels = inputs.pop('labels')

        loss_scale = inputs.pop('loss_scale', None)
        if loss_scale is not None:
            loss_kwargs['loss_scale'] = loss_scale

        outputs = model(**inputs)
        
        loss =  outputs["loss"]
        return (loss, outputs) if return_outputs else loss
        
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is None:
            labels = inputs['labels']
            # fix https://github.com/huggingface/transformers/issues/34263
            if num_items_in_batch is not None:
                outputs.loss = outputs.loss * (labels[:, 1:] != -100).sum() / num_items_in_batch

            if isinstance(outputs, dict) and 'loss' not in outputs:
                raise ValueError(
                    'The model did not return a loss from the inputs, only the following keys: '
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}.")
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs['loss'] if isinstance(outputs, dict) else outputs[0]
        else:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if is_peft_available() and isinstance(unwrapped_model, PeftModel):
                model_name = unwrapped_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            # User-defined compute_loss function
            if self.compute_loss_func is not None:
                loss = self.compute_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch, **loss_kwargs)
            elif model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)

        if self.args.sequence_parallel_size > 1:
            from swift.trainers.xtuner import reduce_xtuner_sequence_parallel_loss
            loss = reduce_xtuner_sequence_parallel_loss(loss, labels)

        if getattr(self.args, 'average_tokens_across_devices', False):
            loss *= self.accelerator.num_processes

        if outputs.logits is not None and labels is not None:
            # Liger does not have logits
            self._compute_acc(outputs, labels)

        return (loss, outputs) if return_outputs else loss
