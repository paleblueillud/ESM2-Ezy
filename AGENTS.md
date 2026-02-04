## ESM-Ezy (ESM2-only) Contributor Notes

- **Backbone:** ESM2 only; default model is `esm2_t36_3B_UR50D`.
- **CUDA/HPC:**
  - Do **not** hardcode `CUDA_VISIBLE_DEVICES`.
  - For single-GPU scripts, use `torch.device("cuda" if torch.cuda.is_available() else "cpu")`.
  - For `torchrun`/distributed scripts, always use `LOCAL_RANK` to select the device (`torch.cuda.set_device(local_rank)`).
  - Assume Linux HPC; avoid macOS-specific paths/tools.
- **Model loading:** `--model_path` accepts either an ESM2 pretrained name (optionally with `()`) or a local `.pt` path.
- **Precision:** default dtype is `float32`; always expose `--dtype` and use autocast when `dtype != float32` on CUDA.
- **Embedding dim:** must be dynamic (no hard-coded `1280`); FAISS index dim must come from computed embeddings.
- **Checkpointing:** save **trainable-only** checkpoint dict (`state_dict` + `meta`); do not save full ESM2 state by default.
- **Repo hygiene:** don’t commit large datasets/weights; `data/` and `ckpt/` are user-managed.
