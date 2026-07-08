"""Train one BatchTopK SAE on a residual-stream hook of LFM2.5-230M."""

import argparse
import json
import os
import pathlib

os.environ.setdefault("WANDB_MODE", "disabled")

from sae_lens import (
    BatchTopKTrainingSAEConfig,
    LanguageModelSAERunnerConfig,
    LanguageModelSAETrainingRunner,
    LoggingConfig,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    p.add_argument("--training-tokens", type=int, required=True)
    p.add_argument("--n-checkpoints", type=int, default=2)
    args = p.parse_args()

    steps = args.training_tokens // 4096
    out_dir = f"/kaggle/working/sae_layer{args.layer}"

    cfg = LanguageModelSAERunnerConfig(
        # NOTE: normalize_activations must stay "none" for BatchTopK. With
        # "expected_average_only_in", SAELens (<=6.45.3) exports a JumpReLU
        # threshold in normalized-activation space while the folded encoder
        # produces raw-space pre-activations (the scaling factor cancels out
        # of W_enc during the decoder-norm fold), so every activation lands
        # below threshold at inference (L0=0). LFM2.5 residual norms are ~1-2
        # vs sqrt(d)=32, making the mismatch a ~20x factor. BatchTopK controls
        # sparsity structurally via k, so normalization is not needed.
        sae=BatchTopKTrainingSAEConfig(
            k=64,
            d_in=1024,
            d_sae=16 * 1024,
            normalize_activations="none",
        ),
        # --- model: HF path, since TransformerLens has no LFM2 support ---
        model_name="LiquidAI/LFM2.5-230M",
        model_class_name="AutoModelForCausalLM",
        hook_name=f"model.layers.{args.layer}",
        model_from_pretrained_kwargs={"dtype": args.dtype},
        # --- data ---
        dataset_path="HuggingFaceFW/fineweb-edu",
        is_dataset_tokenized=False,
        streaming=True,
        context_size=1024,
        prepend_bos=True,
        exclude_special_tokens=True,
        # --- activation store ---
        training_tokens=args.training_tokens,
        store_batch_size_prompts=16,
        n_batches_in_buffer=16,
        prefetch_llm_batches=True,
        # --- optimization ---
        train_batch_size_tokens=4096,
        lr=1e-4,
        lr_scheduler_name="constant",
        lr_warm_up_steps=min(1000, steps // 10),
        lr_decay_steps=steps // 5,
        # --- devices / precision (one visible GPU per process) ---
        device="cuda",
        dtype="float32",
        seed=42,
        # --- outputs ---
        n_checkpoints=args.n_checkpoints,
        checkpoint_path=f"/kaggle/working/checkpoints_layer{args.layer}",
        output_path=out_dir,
        logger=LoggingConfig(log_to_wandb=False),
    )

    print(f"[layer {args.layer}] steps={steps} checkpoints={cfg.checkpoint_path}")
    LanguageModelSAETrainingRunner(cfg).run()

    marker = {"layer": args.layer, "hook": cfg.hook_name, "status": "done"}
    pathlib.Path(out_dir, "DONE.json").write_text(json.dumps(marker))
    print(f"[layer {args.layer}] training complete -> {out_dir}")


if __name__ == "__main__":
    main()
