from collections import OrderedDict
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput

from models import DistilBertForSequenceClassificationSig, XLMRobertaForSequenceClassificationSig


def _normalize_et_model_type(raw_value):
    aliases = {
        "emotion-et": "emotion_et",
    }
    return aliases.get(raw_value or "et2", raw_value or "et2")


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
        et_model_type="et2",
        et_model_id=None,
    ):
        super().__init__()
        self.model_name = model_name
        self.base_model = _build_baseline_model(model_name, checkpoint)
        self.config = self.base_model.config
        self.tokenizer = tokenizer
        self.hidden_size = self.config.hidden_size
        self.num_labels = 2
        self.et_model_type = _normalize_et_model_type(et_model_type)
        self.et_model_id = et_model_id

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

        self.fp_model = self._load_fixation_predictor(et2_checkpoint_path) if load_fixation_model else None
        self.fixation_cache = OrderedDict()
        self.max_fix_cache_size = max_fix_cache_size

    def _load_fixation_predictor(self, et2_checkpoint_path):
        if self.et_model_type == "et2":
            return self._load_et2_predictor(et2_checkpoint_path)
        if self.et_model_type == "emotion_et":
            return self._load_emotion_et_predictor(self.et_model_id or et2_checkpoint_path)
        raise ValueError(f"Unknown et_model_type: {self.et_model_type}")

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

    def _load_emotion_et_predictor(self, et_model_id):
        try:
            from emotion_et_wrapper import EmotionEtFixationsPredictor
        except ImportError as exc:
            raise ImportError(
                "Could not import EmotionEtFixationsPredictor. Install huggingface_hub/safetensors/transformers."
            ) from exc

        fp_model = EmotionEtFixationsPredictor(
            modelTokenizer=self.tokenizer,
            model_id=et_model_id,
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
        et_model_type="et2",
        et_model_id=None,
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
            et_model_type=et_model_type,
            et_model_id=et_model_id,
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
        et_model_type="et2",
        et_model_id=None,
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
            et_model_type=et_model_type,
            et_model_id=et_model_id,
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


class GazeWordConcatForSequenceRegression(GazeConcatForSequenceRegression):
    def _compact_fixations(self, fixations, fixation_attention):
        valid = fixation_attention.bool() & fixations.abs().sum(dim=-1).gt(0)
        lengths = valid.sum(dim=1)
        max_len = int(lengths.max().item()) if lengths.numel() else 0
        if max_len == 0:
            max_len = 1

        batch_size, _, num_features = fixations.shape
        compact = fixations.new_zeros((batch_size, max_len, num_features))
        compact_attention = fixation_attention.new_zeros((batch_size, max_len))

        for row_idx in range(batch_size):
            row_fixations = fixations[row_idx][valid[row_idx]]
            row_len = row_fixations.size(0)
            if row_len:
                compact[row_idx, :row_len] = row_fixations
                compact_attention[row_idx, :row_len] = 1

        return compact, compact_attention

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
        compact_fixations, compact_attention = self._compact_fixations(fixations, fixation_attention)

        compact_projected = self.fixations_embedding_projector(compact_fixations)
        compact_projected = self.norm_layer_fix(compact_projected)

        batch_size = input_ids.size(0)
        eye_start_embed = self.eye_start.to(device=model_device, dtype=text_embeddings.dtype).view(1, 1, -1)
        eye_end_embed = self.eye_end.to(device=model_device, dtype=text_embeddings.dtype).view(1, 1, -1)
        eye_start_embed = eye_start_embed.expand(batch_size, -1, -1)
        eye_end_embed = eye_end_embed.expand(batch_size, -1, -1)
        separator_mask = torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=model_device)

        fused_embeddings = torch.cat(
            (text_embeddings, eye_start_embed, compact_projected, eye_end_embed),
            dim=1,
        )
        extended_attention_mask = torch.cat(
            (attention_mask, separator_mask, compact_attention, separator_mask),
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


class GazePooledHeadForSequenceRegression(GazeBaseForSequenceRegression):
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
        et_model_type="et2",
        et_model_id=None,
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
            et_model_type=et_model_type,
            et_model_id=et_model_id,
        )
        p_1, p_2 = fp_dropout
        pooled_feature_size = len(self.feature_indices) * 3
        self.gaze_pool_projector = nn.Sequential(
            nn.Linear(pooled_feature_size, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(p=p_1),
            nn.Linear(128, self.hidden_size),
            nn.Dropout(p=p_2),
            nn.LayerNorm(self.hidden_size),
        )
        self.pooled_classifier = nn.Linear(self.hidden_size * 2, self.num_labels)

    def _pool_fixations(self, fixations, fixation_attention):
        valid = fixation_attention.bool() & fixations.abs().sum(dim=-1).gt(0)
        valid_f = valid.unsqueeze(-1).to(dtype=fixations.dtype)
        counts = valid_f.sum(dim=1).clamp_min(1.0)

        summed = (fixations * valid_f).sum(dim=1)
        mean = summed / counts

        neg_inf = torch.finfo(fixations.dtype).min
        max_values = fixations.masked_fill(~valid.unsqueeze(-1), neg_inf).max(dim=1).values
        max_values = torch.where(valid.any(dim=1, keepdim=True), max_values, torch.zeros_like(max_values))

        centered = (fixations - mean.unsqueeze(1)) * valid_f
        variance = centered.square().sum(dim=1) / counts
        std = torch.sqrt(variance.clamp_min(0.0))
        return torch.cat((mean, max_values, std), dim=-1)

    def _encode_text_cls(
        self,
        input_ids,
        attention_mask,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        if self.config.model_type == "distilbert":
            outputs = self.base_model.distilbert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )
            cls_repr = outputs.last_hidden_state[:, 0]
            cls_repr = self.base_model.pre_classifier(cls_repr)
            cls_repr = nn.functional.relu(cls_repr)
            cls_repr = self.base_model.dropout(cls_repr)
            return cls_repr, outputs

        outputs = self.base_model.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        cls_repr = outputs.last_hidden_state[:, 0]
        cls_repr = self.base_model.classifier.dropout(cls_repr)
        cls_repr = self.base_model.classifier.dense(cls_repr)
        cls_repr = torch.tanh(cls_repr)
        cls_repr = self.base_model.classifier.dropout(cls_repr)
        return cls_repr, outputs

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

        cls_repr, outputs = self._encode_text_cls(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        fixations, fixation_attention = self._compute_fixations_batch(input_ids, attention_mask)
        fixations = fixations.to(device=model_device, dtype=cls_repr.dtype)
        fixation_attention = fixation_attention.to(device=model_device, dtype=attention_mask.dtype)
        pooled_fixations = self._pool_fixations(fixations, fixation_attention)
        gaze_repr = self.gaze_pool_projector(pooled_fixations)

        logits = self.pooled_classifier(torch.cat((cls_repr, gaze_repr), dim=-1))
        logits = self.base_model.sigmoid(logits)

        return SequenceClassifierOutput(
            loss=None,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
