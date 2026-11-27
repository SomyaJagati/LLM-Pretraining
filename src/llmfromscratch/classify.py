import sys
import torch
import tiktoken

# For local (use `from LLMArchitecture import LLM` on Colab)
from llmfromscratch.LLMArchitecture import LLM
from llmfromscratch.LoRA import LinearWithLoRA


# ---------- Settings: these must match how you trained ----------
CHECKPOINT = r"D:\Projects\LLMFromScratch\models\classification_finetuned_model.pth"
MAX_LENGTH = 128            # same max_length you used in the training dataset
NUM_CLASSES = 2
LORA_RANK = 16
LORA_ALPHA = 16
PAD_TOKEN_ID = 50256
LABEL_NAMES = {0: "negative", 1: "positive"}

CONFIG = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 384,
    "n_heads": 6,
    "n_layers": 6,
    "drop_rate": 0.2,
    "qkv_bias": False,
}
# -----------------------------------------------------------------


def replace_linear_with_lora(model, rank, alpha):
    """Same function as in the training notebook."""
    for name, module in model.named_children():
        if isinstance(module, torch.nn.MultiheadAttention):
            continue
        if isinstance(module, torch.nn.Linear):
            setattr(model, name, LinearWithLoRA(module, rank, alpha))
        else:
            replace_linear_with_lora(module, rank, alpha)


def last_token_logits(model, input_batch, pad_id=PAD_TOKEN_ID):
    """Same function as in training: read the output at the last real (non-pad) token."""
    logits = model(input_batch)
    lengths = (input_batch != pad_id).sum(dim=1)
    idx = (lengths - 1).clamp(min=0)
    return logits[torch.arange(logits.size(0), device=logits.device), idx, :]


def load_model(checkpoint_path, device):
    """Rebuild the exact training architecture, then load the finetuned weights."""
    model = LLM(CONFIG)
    model.out_head = torch.nn.Linear(CONFIG["emb_dim"], NUM_CLASSES)   # classification head
    replace_linear_with_lora(model, rank=LORA_RANK, alpha=LORA_ALPHA)  # same order as training

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model


def classify_sentiment(text, model, tokenizer, device, max_length=MAX_LENGTH, pad_token_id=PAD_TOKEN_ID):
    """Return (label, confidence) for one piece of text."""
    model.eval()

    supported_context_length = model.pos_emb.weight.shape[0]
    assert max_length <= supported_context_length, (
        f"max_length ({max_length}) exceeds the model's context length ({supported_context_length})."
    )

    # Tokenize and truncate
    input_ids = tokenizer.encode(text)[:max_length]
    if len(input_ids) == 0:
        raise ValueError("The text is empty after tokenization.")

    # Pad to max_length, exactly as the training dataset did
    input_ids += [pad_token_id] * (max_length - len(input_ids))
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)  # add batch dimension

    with torch.no_grad():
        logits = last_token_logits(model, input_tensor, pad_token_id)   # (1, num_classes)

    probs = torch.softmax(logits, dim=-1)[0]
    predicted = int(torch.argmax(probs).item())
    return LABEL_NAMES[predicted], probs[predicted].item()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = tiktoken.get_encoding("gpt2")
    model = load_model(CHECKPOINT, device)

    if len(sys.argv) > 1:
        texts = [" ".join(sys.argv[1:])]
    else:
        texts = [
            "O, what a joy it is to see thee well, my dearest friend!",
            "Thou art a villain, and I curse the day that gave thee birth.",
            "The king hath gone to the city and will return by noon.",
        ]

    for text in texts:
        label, confidence = classify_sentiment(text, model, tokenizer, device)
        print(f"{label} ({confidence:.1%}): {text}")