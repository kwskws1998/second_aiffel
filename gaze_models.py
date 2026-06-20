from collections import OrderedDict
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput

from models import DistilBertForSequenceClassificationSig, XLMRobertaForSequenceClassificationSig


def _build_baseline_model(model_name, checkpoint):
    if model_name == "distilbert":
        return DistilBertForSequenceClassificationSig.from_pretrained(checkpoint, num_labels=2)
    if model_name in ("xlmroberta-base", "xlmroberta-large"):
        return XLMRobertaForSequenceClassificationSig.from_pretrained(checkpoint, num_labels=2)
    raise ValueError(f"Unknown model name: {model_name}")


class GazeBaseForSequenceRegression(nn.Module):
    def __init__(
        self,
        model_name,
        checkpoint,
        tokenizer,
        et2_checkpoint_path=None,
        features_used=None,
        fp_dropout=(0.0, 0.3),
        max_fix_cache_size=20000,
        load_fixation_model=True,
    ):
        super().__init__()
        self.model_name = model_name
        self.base_model = _build_baseline_model(model_name, checkpoint)
        self.config = self.base_model.config
        self.tokenizer = tokenizer
        self.hidden_size = self.config.hidden_size
        self.num_labels = 2

        flags = features_used or [1, 1, 1, 1, 1]
        self.feature_indices = [idx for idx, enabled in enumerate(flags) if int(enabled) == 1]
        if not self.feature_indices:
            raise ValueError("features_used must enable at least one ET feature.")

        p_1, p_2 = fp_dropout
        self.fixations_embedding_projector = nn.Sequential(
            nn.Linear(len(self.feature_indices), 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(p=p_1),
            nn.Linear(128, self.hidden_size),
            nn.Dropout(p=p_2),
        )
        self.norm_layer_fix = nn.LayerNorm(self.hidden_size)

        self.fp_model = self._load_et2_predictor(et2_checkpoint_path) if load_fixation_model else None
        self.fixation_cache = OrderedDict()
        self.max_fix_cache_size = max_fix_cache_size

    def _load_et2_predictor(self, et2_checkpoint_path):
        try:
            from et2_wrapper import FixationsPredictor_2
        except ImportError as exc:
            raise ImportError(
                "Could not import FixationsPredictor_2. Check et2_wrapper.py and the ET2 checkpoint path."
            ) from exc

        fp_model = FixationsPredictor_2(
            modelTokenizer=self.tokenizer,
            remap=False,
            checkpoint_path=et2_checkpoint_path,
        )
        if hasattr(fp_model, "model"):
            fp_model.model.eval()
            for param in fp_model.model.parameters():
                param.requires_grad = False
        return fp_model

    @staticmethod
    def _build_cache_key(token_ids_1d, attention_mask_1d):
        valid_len = int(attention_mask_1d.sum().item())
        if valid_len <= 0:
            return tuple(), valid_len
        return tuple(token_ids_1d[:valid_len].tolist()), valid_len

    def _empty_fixations(self, seq_len, dtype, device):
        return torch.zeros(seq_len, len(self.feature_indices), dtype=dtype, device=device)

    def _predict_fixations_single(self, token_ids_1d, attention_mask_1d):
        device = token_ids_1d.device
        seq_len = token_ids_1d.shape[0]
        key, valid_len = self._build_cache_key(token_ids_1d, attention_mask_1d)

        if valid_len <= 0:
            return (
                self._empty_fixations(seq_len, torch.float32, device),
                torch.zeros(seq_len, dtype=attention_mask_1d.dtype, device=device),
            )

        if self.fp_model is None:
            return (
                self._empty_fixations(seq_len, torch.float32, device),
                attention_mask_1d.to(device=device),
            )

        cached = self.fixation_cache.get(key)
        if cached is None:
            sample_ids = token_ids_1d[:valid_len].unsqueeze(0)
            sample_mask = attention_mask_1d[:valid_len].unsqueeze(0)
            with torch.no_grad():
                fixations, fixation_mask, _, _, _, _ = self.fp_model._compute_mapped_fixations(
                    sample_ids, sample_mask
                )

            fixations = fixations.squeeze(0).float().cpu()
            fixation_mask = fixation_mask.squeeze(0).long().cpu()
            fixations = fixations[:, self.feature_indices]

            if len(self.fixation_cache) >= self.max_fix_cache_size:
                self.fixation_cache.popitem(last=False)
            self.fixation_cache[key] = (fixations, fixation_mask)
        else:
            fixations, fixation_mask = cached
            self.fixation_cache.move_to_end(key)

        fixations = fixations.to(device)
        fixation_mask = fixation_mask.to(device=device, dtype=attention_mask_1d.dtype)

        padded_fixations = self._empty_fixations(seq_len, fixations.dtype, device)
        padded_mask = torch.zeros(seq_len, dtype=attention_mask_1d.dtype, device=device)
        copy_len = min(valid_len, fixations.shape[0], seq_len)
        padded_fixations[:copy_len] = fixations[:copy_len]
        padded_mask[:copy_len] = fixation_mask[:copy_len].to(dtype=attention_mask_1d.dtype)
        return padded_fixations, padded_mask

    def _compute_fixations_batch(self, input_ids, attention_mask):
        batch_fixations = []
        batch_masks = []
        for row_idx in range(input_ids.size(0)):
            row_fix, row_mask = self._predict_fixations_single(
                input_ids[row_idx], attention_mask[row_idx]
            )
            batch_fixations.append(row_fix)
            batch_masks.append(row_mask)
        return torch.stack(batch_fixations, dim=0), torch.stack(batch_masks, dim=0)

    def _encode_base(
        self,
        attention_mask,
        inputs_embeds,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        kwargs = {
            "input_ids": None,
            "attention_mask": attention_mask,
            "inputs_embeds": inputs_embeds,
            "labels": None,
            "output_attentions": output_attentions,
            "output_hidden_states": output_hidden_states,
            "return_dict": return_dict,
        }
        if self.config.model_type != "distilbert":
            kwargs["token_type_ids"] = token_type_ids
            kwargs["position_ids"] = position_ids
        if head_mask is not None:
            kwargs["head_mask"] = head_mask
        return self.base_model(**kwargs)


class GazeAddForSequenceRegression(GazeBaseForSequenceRegression):
    def __init__(
        self,
        model_name,
        checkpoint,
        tokenizer,
        et2_checkpoint_path=None,
        features_used=None,
        fp_dropout=(0.0, 0.3),
        max_fix_cache_size=20000,
        gaze_add_scale=0.05,
        train_gaze_add_scale=False,
        load_fixation_model=True,
    ):
        skip_fixed_zero_gaze = not train_gaze_add_scale and float(gaze_add_scale) == 0.0
        super().__init__(
            model_name=model_name,
            checkpoint=checkpoint,
            tokenizer=tokenizer,
            et2_checkpoint_path=et2_checkpoint_path,
            features_used=features_used,
            fp_dropout=fp_dropout,
            max_fix_cache_size=max_fix_cache_size,
            load_fixation_model=load_fixation_model and not skip_fixed_zero_gaze,
        )
        self.skip_fixed_zero_gaze = skip_fixed_zero_gaze
        gaze_add_scale = torch.tensor(float(gaze_add_scale))
        if train_gaze_add_scale:
            self.gaze_add_scale = nn.Parameter(gaze_add_scale)
        else:
            self.register_buffer("gaze_add_scale", gaze_add_scale)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        if input_ids is None:
            raise ValueError("input_ids cannot be None.")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        embed_layer = self.base_model.get_input_embeddings()
        model_device = embed_layer.weight.device
        input_ids = input_ids.to(model_device)
        attention_mask = attention_mask.to(model_device)

        text_embeddings = embed_layer(input_ids)
        if self.skip_fixed_zero_gaze:
            fused_embeddings = text_embeddings
        else:
            fixations, _ = self._compute_fixations_batch(input_ids, attention_mask)
            fixations = fixations.to(device=model_device, dtype=text_embeddings.dtype)
            fixations_projected = self.fixations_embedding_projector(fixations)
            fixations_projected = self.norm_layer_fix(fixations_projected)
            gaze_present = fixations.abs().sum(dim=-1, keepdim=True).gt(0).to(dtype=text_embeddings.dtype)
            fused_embeddings = text_embeddings + self.gaze_add_scale * fixations_projected * gaze_present

        return self._encode_base(
            attention_mask=attention_mask,
            inputs_embeds=fused_embeddings,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class GazeConcatForSequenceRegression(GazeBaseForSequenceRegression):
    def __init__(
        self,
        model_name,
        checkpoint,
        tokenizer,
        et2_checkpoint_path=None,
        features_used=None,
        fp_dropout=(0.0, 0.3),
        max_fix_cache_size=20000,
        load_fixation_model=True,
    ):
        super().__init__(
            model_name=model_name,
            checkpoint=checkpoint,
            tokenizer=tokenizer,
            et2_checkpoint_path=et2_checkpoint_path,
            features_used=features_used,
            fp_dropout=fp_dropout,
            max_fix_cache_size=max_fix_cache_size,
            load_fixation_model=load_fixation_model,
        )
        self.eye_start = nn.Parameter(torch.zeros(self.hidden_size))
        self.eye_end = nn.Parameter(torch.zeros(self.hidden_size))

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        if input_ids is None:
            raise ValueError("input_ids cannot be None.")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        embed_layer = self.base_model.get_input_embeddings()
        model_device = embed_layer.weight.device
        input_ids = input_ids.to(model_device)
        attention_mask = attention_mask.to(model_device)

        text_embeddings = embed_layer(input_ids)
        fixations, fixation_attention = self._compute_fixations_batch(input_ids, attention_mask)
        fixations = fixations.to(device=model_device, dtype=text_embeddings.dtype)
        fixation_attention = fixation_attention.to(device=model_device, dtype=attention_mask.dtype)

        fixations_projected = self.fixations_embedding_projector(fixations)
        fixations_projected = self.norm_layer_fix(fixations_projected)

        batch_size = input_ids.size(0)
        eye_start_embed = self.eye_start.to(device=model_device, dtype=text_embeddings.dtype).view(1, 1, -1)
        eye_end_embed = self.eye_end.to(device=model_device, dtype=text_embeddings.dtype).view(1, 1, -1)
        eye_start_embed = eye_start_embed.expand(batch_size, -1, -1)
        eye_end_embed = eye_end_embed.expand(batch_size, -1, -1)
        separator_mask = torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=model_device)

        fused_embeddings = torch.cat(
            (text_embeddings, eye_start_embed, fixations_projected, eye_end_embed),
            dim=1,
        )
        extended_attention_mask = torch.cat(
            (attention_mask, separator_mask, fixation_attention, separator_mask),
            dim=1,
        )

        return self._encode_base(
            attention_mask=extended_attention_mask,
            inputs_embeds=fused_embeddings,
            token_type_ids=None,
            position_ids=None,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
