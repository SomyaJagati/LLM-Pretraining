import tiktoken
import torch

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # remove batch dimension
    return tokenizer.decode(flat.tolist())

def generate_text_simple(model, idx, max_new_tokens, context_size):
    """Based on higest probablity token."""
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):

        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on the last time step
        # (batch, n_token, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]

        # Get the idx of the vocab entry with the highest logits value
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx

def generate_text(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):

    # For-loop is the same as before: Get logits, and only focus on last time step
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        # New: Filter logits with top_k sampling
        if top_k is not None:
            # Keep only top_k values
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1:]   # keepdim so it broadcasts per row
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

        # New: Apply temperature scaling
        if temperature > 0.0:
            logits = logits / temperature

            # New (not in book): numerical stability tip to get equivalent results on mps device
            # subtract rowwise max before softmax
            logits = logits - logits.max(dim=-1, keepdim=True).values

            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)  # (batch_size, vocab_size)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        # Otherwise same as before: get idx of the vocab entry with the highest logits value
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        if eos_id is not None and idx_next.item() == eos_id:  # Stop early if end-of-sequence token is encountered
            break

        # Same as before: append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    return idx


if __name__ == "__main__":
    from llmfromscratch.LLMArchitecture import LLM

    #CONGIGURATION (must match the training config)
    CONFIG = {
        "vocab_size": 50257,   # Vocabulary size
        "context_length": 256, # Shortened context length (orig: 1024)
        "emb_dim": 384,        # Embedding dimension
        "n_heads": 6,          # Number of attention heads
        "n_layers": 6,         # Number of layers
        "drop_rate": 0.2,      # Dropout rate
        "qkv_bias": False      # Query-key-value bias
    }

    #setting the device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #Tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    #Loading the model
    model = LLM(CONFIG)
    ckpt = torch.load(r"D:\Projects\LLMFromScratch\models\final_pretrained_model.pth", map_location=device, weights_only=True)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    #Generate text
    prompt = "First Citizen:"
    encoded = text_to_token_ids(prompt, tokenizer).to(device)

    #Greedy
    torch.manual_seed(42)
    out = generate_text(model, encoded, max_new_tokens=100, context_size=CONFIG["context_length"],
                        temperature=0.0, top_k=None)
    print("--- GREEDY ---")
    print(token_ids_to_text(out, tokenizer))

    #Sampled
    out = generate_text(model, encoded, max_new_tokens=100, context_size=CONFIG["context_length"],
                        temperature=0.8, top_k=40)
    print("\n--- SAMPLED (temp 0.8, top-k 40) ---")
    print(token_ids_to_text(out, tokenizer))