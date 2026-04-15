import os

import modal
from cs336_basics.modal_utils import DATA_PATH, VOLUME_MOUNTS, app, build_image

wandb_secret = modal.Secret.from_name("wandb")


@app.function(
    image=build_image(),
    volumes=VOLUME_MOUNTS,
    gpu="B200",
    secrets=[wandb_secret],
    timeout=600,
)
def generate():
    from src.training_helpers import AdamW, decode_lm, load_checkpoint
    from src.transformer import TransformerLM, RotaryPositionalEmbedding
    import torch
    from src.tokenizer import Tokenizer

    device = "cuda"

    rope = RotaryPositionalEmbedding(10000.0, 512 // 16, max_seq_len=256)
    lm = TransformerLM(
        d_model=512,
        num_heads=16,
        d_ff=1344,
        vocab_size=10000,
        context_length=256,
        num_layers=4,
        rope=rope,
    )
    lm.to(device=device)
    optimizer = AdamW(lm.parameters(), 0.008, [0.9, 0.999], 1e-8, 0.01)
    data = torch.load(str(DATA_PATH / "checkpoints" / "bs_256.pt"))
    no_compile = {k.replace("_orig_mod.", ""): v for k, v in data["model_state"].items()}
    lm.load_state_dict(no_compile)
    optimizer.load_state_dict(data["optimizer_state"])

    tokenizer = Tokenizer.from_files(
        str(DATA_PATH / "output_tiny_bpe" / "vocab.pkl"),
        str(DATA_PATH / "output_tiny_bpe" / "merges.pkl"),
        ["<|endoftext|>"],
    )

    prompt = "Once upon a time"
    prompt_tokens = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    temperatures = [0.5, 0.6, 0.7, 0.8]
    top_ps = [0.8, 0.85, 0.9, 0.95]
    for temp in temperatures:
        for top_p in top_ps:
            text = decode_lm(lm, prompt_tokens, 256, temp, top_p, tokenizer)
            print(f"Temperature: {temp}; top_p: {top_p}")
            print(f"Once upon a time {text}")
    return


@app.local_entrypoint()
def main():
    generate.remote()
