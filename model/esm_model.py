import os

import esm
import torch
import torch.nn as nn

_DTYPE_NAME_MAP = {
    "float32": "float32",
    "fp32": "float32",
    "float": "float32",
    "float16": "float16",
    "fp16": "float16",
    "half": "float16",
    "bfloat16": "bfloat16",
    "bf16": "bfloat16",
}

_DTYPE_TORCH_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def normalize_dtype_name(dtype_name):
    if dtype_name is None:
        return "float32"
    key = str(dtype_name).strip().lower()
    if key in _DTYPE_NAME_MAP:
        return _DTYPE_NAME_MAP[key]
    raise ValueError(f"Unsupported dtype '{dtype_name}'. Use float32, float16, or bfloat16.")


def resolve_dtype(dtype_name):
    normalized = normalize_dtype_name(dtype_name)
    return _DTYPE_TORCH_MAP[normalized]


def _normalize_model_name(name):
    clean = str(name).strip()
    if clean.endswith("()"):
        clean = clean[:-2]
    return clean


def _load_esm_model(pretrained_model_path):
    model_path = str(pretrained_model_path).strip()
    if os.path.exists(model_path):
        model, alphabet = esm.pretrained.load_model_and_alphabet_local(model_path)
        return model, alphabet, model_path
    model_name = _normalize_model_name(model_path)
    if not hasattr(esm.pretrained, model_name):
        raise ValueError(
            f"Unknown ESM model '{pretrained_model_path}'. Provide an ESM2 model name "
            f"(e.g., esm2_t36_3B_UR50D) or a local .pt path."
        )
    model_fn = getattr(esm.pretrained, model_name)
    model, alphabet = model_fn()
    return model, alphabet, model_name


def _infer_repr_dim(model):
    if hasattr(model, "embed_dim"):
        return int(model.embed_dim)
    if hasattr(model, "args") and hasattr(model.args, "embed_dim"):
        return int(model.args.embed_dim)
    if hasattr(model, "embed_tokens") and hasattr(model.embed_tokens, "embedding_dim"):
        return int(model.embed_tokens.embedding_dim)
    raise ValueError("Unable to infer embedding dimension from the ESM model.")


class LaccaseModel(nn.Module):
    def __init__(self, pretrained_model_path, dtype="float32"):
        super(LaccaseModel, self).__init__()
        self.modelEsm, alphabet, base_model = _load_esm_model(pretrained_model_path)
        self.base_model = base_model
        self.converter = alphabet.get_batch_converter()
        self.repr_dim = _infer_repr_dim(self.modelEsm)
        self.dnn = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.repr_dim, 2)
        )

        self._device = None
        self._dtype_name = normalize_dtype_name(dtype)
        
    @property
    def device(self):
        if self._device is None:
            self._device = next(self.modelEsm.parameters()).device
        return self._device

    def forward(self, data, return_repr=False):
        out_result = self._get_representations(data)
        out_put = self.dnn(out_result)
        if return_repr:
            return out_put, out_result
        else:
            return out_put
    
    def _get_layers(self):
        return len(self.modelEsm.layers)
    
    @property
    def layers(self):
        return self.get_layers()
    
    def get_layers(self):
        return self._get_layers()
    
    def get_last_layer_idx(self):
        return self._get_layers()-1
    
    
    def _get_representations(self, data):
        names, sequences, tokens = self.converter(data)
        if self.device is not None:
            tokens = tokens.to(self.device)
        # truncate tokens to max 1022 tokens
        tokens = tokens[:, :1022]
        # get the last layer representations
        last_layer_idx = self._get_layers()
        result = self.modelEsm(tokens, repr_layers=[last_layer_idx])
        out_result = result["representations"][last_layer_idx][:, 0, :]
        return out_result
    
    def get_representations(self, data):
        return self._get_representations(data)
    
    def get_names(self, data):
        names, sequences, tokens = self.converter(data)
        return names

    @property
    def dtype_name(self):
        return self._dtype_name
    
    @classmethod
    def from_pretrained(cls, pretrained_model_path, state_dict_path=None, dtype="float32", map_location="cpu"):
        model = cls(pretrained_model_path, dtype=dtype)
        if state_dict_path is not None:
            print(f"Loading state dict from {state_dict_path}")
            checkpoint = torch.load(state_dict_path, map_location=map_location)
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
                meta = checkpoint.get("meta", {})
                if "repr_dim" in meta and int(meta["repr_dim"]) != model.repr_dim:
                    raise ValueError(
                        f"Checkpoint repr_dim {meta['repr_dim']} does not match "
                        f"current model repr_dim {model.repr_dim}."
                    )
            else:
                state_dict = checkpoint
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"Warning: missing keys when loading state dict: {len(missing)}")
            if unexpected:
                print(f"Warning: unexpected keys when loading state dict: {len(unexpected)}")
        return model

        

