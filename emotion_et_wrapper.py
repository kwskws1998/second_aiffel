"""Hugging Face emotion-specific ET predictor adapter for VA gaze fusion."""

import os
import re
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer, RobertaConfig, RobertaModel

try:
    from huggingface_hub import snapshot_download
except ImportError:
    snapshot_download = None


DEFAULT_REPO_ID = "skboy/emotion_et_2nd_model"
DEFAULT_WEIGHT_NAME = "et_predictor2_iitb_sa1_sa2_lr2e5_len256_seed123.safetensors"
FEATURE_NAMES = ["nFix", "FFD", "GPT", "TRT", "fixProp"]
WINDOW_SIZE = 512
MODEL_SUBDIR_ENV = "EMOTION_ET_MODEL_SUBDIR"


class _EmotionEtRegressionModel(torch.nn.Module):
    def __init__(self, config_path):
        super().__init__()
        config = RobertaConfig.from_pretrained(config_path)
        self.roberta = RobertaModel(config)
        self.decoder = torch.nn.Linear(config.hidden_size, len(FEATURE_NAMES))

    def forward(self, input_ids, attention_mask):
        hidden = self.roberta(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.decoder(hidden)


class EmotionEtFixationsPredictor:
    """Adapter exposing the same _compute_mapped_fixations interface as ET2."""

    def __init__(self, modelTokenizer, model_id=None, weight_name=None, max_length=WINDOW_SIZE, device=None):
        self.rm_tokenizer = modelTokenizer
        self.model_id = model_id or os.environ.get("EMOTION_ET_MODEL_ID") or DEFAULT_REPO_ID
        self.weight_name = weight_name or os.environ.get("EMOTION_ET_WEIGHT_NAME") or DEFAULT_WEIGHT_NAME
        self.max_length = int(max_length)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model_dir = self._resolve_model_dir(self.model_id)
        self.roberta_tokenizer = AutoTokenizer.from_pretrained(self.model_dir, add_prefix_space=True)
        self.model = _EmotionEtRegressionModel(self.model_dir).to(self.device)
        self._load_weights()
        self.model.eval()
        self.feature_names = FEATURE_NAMES
        self.feature_dim = len(FEATURE_NAMES)
        print(f"[emotion_et_wrapper] EmotionEtFixationsPredictor loaded: {self.model_dir}")

    def _resolve_model_dir(self, model_id):
        candidate = Path(model_id).expanduser()
        if candidate.exists():
            return self._select_model_dir(candidate)
        if snapshot_download is None:
            raise ImportError("huggingface_hub is required to download an emotion ET model.")
        local_files_only = os.environ.get("EMOTION_ET_LOCAL_FILES_ONLY", "").lower() in {
            "1",
            "true",
            "yes",
        }
        snapshot_dir = Path(snapshot_download(model_id, local_files_only=local_files_only))
        return self._select_model_dir(snapshot_dir)

    def _select_model_dir(self, base_dir):
        subdir = os.environ.get(MODEL_SUBDIR_ENV)
        if subdir:
            model_dir = base_dir / subdir
            if not self._is_model_dir(model_dir):
                raise FileNotFoundError(
                    f"{MODEL_SUBDIR_ENV}={subdir} does not point to a valid emotion ET model under {base_dir}."
                )
            return model_dir

        if self._is_model_dir(base_dir):
            return base_dir

        candidates = sorted(
            {
                path.parent
                for path in base_dir.rglob("config.json")
                if self._is_model_dir(path.parent)
            },
            key=lambda path: str(path.relative_to(base_dir)),
        )
        if len(candidates) == 1:
            return candidates[0]

        named_weight_matches = [path for path in candidates if (path / self.weight_name).exists()]
        if len(named_weight_matches) == 1:
            return named_weight_matches[0]

        if candidates:
            candidate_list = ", ".join(str(path.relative_to(base_dir)) for path in candidates)
            raise FileNotFoundError(
                f"Multiple emotion ET model directories found under {base_dir}: {candidate_list}. "
                f"Set {MODEL_SUBDIR_ENV} to choose one."
            )
        raise FileNotFoundError(
            f"No emotion ET model directory found under {base_dir}. Expected config.json and one .safetensors file."
        )

    @staticmethod
    def _is_model_dir(path):
        path = Path(path)
        return path.is_dir() and (path / "config.json").exists() and any(path.glob("*.safetensors"))

    def _load_weights(self):
        weight_path = self.model_dir / self.weight_name
        if not weight_path.exists():
            safetensors_files = sorted(self.model_dir.glob("*.safetensors"))
            if len(safetensors_files) == 1:
                weight_path = safetensors_files[0]
            else:
                raise FileNotFoundError(f"Emotion ET weight not found: {self.model_dir / self.weight_name}")
        state = load_file(str(weight_path), device=str(self.device))
        self.model.load_state_dict(state, strict=True)
        print(f"[emotion_et_wrapper] weights loaded: {weight_path}")

    def _compute_mapped_fixations(self, input_ids_rm, attention_mask_rm=None):
        if attention_mask_rm is None:
            attention_mask_rm = torch.ones_like(input_ids_rm)

        ids = input_ids_rm[0].detach().cpu().tolist()
        mask = attention_mask_rm[0].detach().cpu().tolist()
        pad_id = self.rm_tokenizer.pad_token_id or 0
        ids_no_pad = [i for i, m in zip(ids, mask) if m == 1 and i != pad_id]
        text = self.rm_tokenizer.decode(ids_no_pad, skip_special_tokens=True)

        word_features, words = self._predict_words(text)
        remapped = self._remap_to_rm_tokens(word_features, words, ids, mask)
        fixations = remapped.unsqueeze(0).to(input_ids_rm.device)
        fix_attn = torch.tensor(mask, dtype=torch.long).unsqueeze(0).to(input_ids_rm.device)
        return fixations, fix_attn, None, None, None, None

    def _predict_words(self, text):
        words = self._segment_text(text)
        if not words:
            return np.zeros((0, self.feature_dim), dtype=np.float32), words

        encoded = self.roberta_tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        word_ids = encoded.word_ids(batch_index=0)

        with torch.no_grad():
            token_preds = self.model(input_ids=input_ids, attention_mask=attention_mask)
        token_preds = token_preds.squeeze(0).clamp_min(0.0).detach().cpu().numpy()

        word_features = np.zeros((len(words), self.feature_dim), dtype=np.float32)
        seen = set()
        for token_idx, word_idx in enumerate(word_ids):
            if word_idx is None or word_idx in seen or word_idx >= len(words):
                continue
            word_features[word_idx] = token_preds[token_idx]
            seen.add(word_idx)
        return word_features, words

    @staticmethod
    def _is_cjk(ch):
        code = ord(ch)
        return (
            0x4E00 <= code <= 0x9FFF
            or 0x3040 <= code <= 0x30FF
            or 0xAC00 <= code <= 0xD7AF
        )

    @classmethod
    def _segment_text(cls, text):
        text = (text or "").strip()
        if not text:
            return []
        if any(ch.isspace() for ch in text):
            words = text.split()
            if words:
                return words
        if any(cls._is_cjk(ch) for ch in text):
            return [ch for ch in text if not ch.isspace()]
        return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)

    def _remap_to_rm_tokens(self, word_features, words, rm_ids, rm_mask):
        seq_len = len(rm_ids)
        output = torch.zeros(seq_len, self.feature_dim, dtype=torch.float32)
        if len(word_features) == 0 or len(words) == 0:
            return output

        rm_tokens = self.rm_tokenizer.convert_ids_to_tokens(rm_ids)
        word_to_rm = _align_words_to_rm_tokens(words, rm_tokens, self.rm_tokenizer)

        n_words = min(len(words), len(word_features))
        for word_idx in range(n_words):
            if word_idx >= len(word_to_rm):
                break
            indices = word_to_rm[word_idx]
            if not indices:
                continue
            first = indices[0]
            if first < seq_len and rm_mask[first] == 1:
                output[first] = torch.tensor(word_features[word_idx], dtype=torch.float32)
        return output


def _align_words_to_rm_tokens(words, rm_tokens, rm_tokenizer):
    special_ids = set(rm_tokenizer.all_special_ids)
    word_to_indices = []
    token_idx = 0

    for word in words:
        indices = []
        chars_remaining = len(_normalize_for_alignment(word))
        while token_idx < len(rm_tokens) and chars_remaining > 0:
            token = rm_tokens[token_idx]
            token_id = rm_tokenizer.convert_tokens_to_ids(token)
            if token_id in special_ids:
                token_idx += 1
                continue

            token_clean = _normalize_for_alignment(token.lstrip("Ġ▁ "))
            if token_clean:
                indices.append(token_idx)
                chars_remaining -= len(token_clean)
            token_idx += 1
        word_to_indices.append(indices)
    return word_to_indices


def _normalize_for_alignment(text):
    text = str(text).lstrip("Ġ▁ ")
    if text.startswith("##"):
        text = text[2:]
    return re.sub(r"\s+", "", text)
