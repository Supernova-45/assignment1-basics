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
    from cs336_basics.training_helpers import AdamW, decode_lm, load_checkpoint
    from cs336_basics.transformer import TransformerLM, RotaryPositionalEmbedding
    import torch
    from cs336_basics.tokenizer import Tokenizer

    device = "cuda"

    rope = RotaryPositionalEmbedding(10000.0, 768 // 12, max_seq_len=512)
    lm = TransformerLM(
        d_model=768,
        num_heads=12,
        d_ff=3072,
        vocab_size=32000,
        context_length=512,
        num_layers=6,
        tie_embeddings=True,
        use_relu=True,
        rope=rope,
    )
    lm.to(device=device)
    optimizer = AdamW(lm.parameters(), 0.003, [0.9, 0.95], 1e-8, 0.1)
    load_checkpoint(str(DATA_PATH / "checkpoints" / "test_test_112.pt"), lm, optimizer)

    tokenizer = Tokenizer.from_files(
        str(DATA_PATH / "output_owt_bpe" / "vocab.pkl"),
        str(DATA_PATH / "output_owt_bpe" / "merges.pkl"),
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
            print(f"Once upon a time{text}")
    return


@app.local_entrypoint()
def main():
    generate.remote()
