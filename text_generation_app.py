import os
import sys
import time
from pathlib import Path
import tiktoken
import torch
import streamlit as st

# Ensure src is on python path
src_dir = Path(__file__).resolve().parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from llmfromscratch.LLMArchitecture import LLM

# Model Configuration (matches final_pretrained_model.pth)
MODEL_CONFIG = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 384,
    "n_heads": 6,
    "n_layers": 6,
    "drop_rate": 0.2,
    "qkv_bias": False,
}

CHECKPOINT_PATH = Path(__file__).resolve().parent / "models" / "final_pretrained_model.pth"

st.set_page_config(
    page_title="LLM From Scratch - Text Generation",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (professional, no emojis)
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 0.95rem;
        color: #6c757d;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 6px;
        padding: 12px 16px;
        text-align: center;
    }
    .metric-value {
        font-size: 1.3rem;
        font-weight: 600;
        color: #212529;
    }
    .metric-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #6c757d;
    }
    .output-container {
        background-color: #fdfdfd;
        border: 1px solid #dee2e6;
        border-radius: 6px;
        padding: 18px;
        font-family: inherit;
        line-height: 1.6;
        white-space: pre-wrap;
        min-height: 180px;
    }
    .badge {
        display: inline-block;
        padding: 3px 8px;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 4px;
        background-color: #e9ecef;
        color: #495057;
        margin-right: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_llm_model(checkpoint_path: Path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LLM(MODEL_CONFIG)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    tokenizer = tiktoken.get_encoding("gpt2")
    return model, tokenizer, device


def stream_generation(
    model,
    encoded_prompt: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
    temperature: float,
    top_k: int | None,
    tokenizer,
    device: str,
):
    """Yields each generated token string one by one for streaming."""
    idx = encoded_prompt.to(device)
    eos_id = 50256  # <|endoftext|>

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        if top_k is not None and top_k > 0:
            top_k_clamped = min(top_k, logits.shape[-1])
            top_logits, _ = torch.topk(logits, top_k_clamped)
            min_val = top_logits[:, -1:]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float("-inf"), device=logits.device),
                logits,
            )

        if temperature > 0.0:
            logits = logits / temperature
            logits = logits - logits.max(dim=-1, keepdim=True).values
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        token_id = idx_next.item()
        if token_id == eos_id:
            break

        idx = torch.cat((idx, idx_next), dim=1)
        token_str = tokenizer.decode([token_id])
        yield token_str, token_id


# Initialize model
try:
    model, tokenizer, device = load_llm_model(CHECKPOINT_PATH)
    model_loaded = True
except Exception as exc:
    model_loaded = False
    load_error_msg = str(exc)

# Sidebar Configuration
with st.sidebar:
    st.subheader("Generation Settings")

    strategy = st.radio(
        label="Generation Approach",
        options=["Top-K + Temperature Sampling", "Greedy Approach"],
        index=0,
        help="Choose between probabilistic sampling or deterministic greedy argmax decoding.",
    )

    st.markdown("---")
    st.subheader("Hyperparameters")

    if strategy == "Top-K + Temperature Sampling":
        temperature = st.slider(
            label="Temperature",
            min_value=0.1,
            max_value=2.0,
            value=0.8,
            step=0.05,
            help="Higher values scale logits to increase diversity. Lower values concentrate probability on top candidates.",
        )
        top_k = st.slider(
            label="Top-K",
            min_value=1,
            max_value=100,
            value=40,
            step=1,
            help="Limits candidate tokens to the K highest probability choices at each generation step.",
        )
    else:
        st.caption(
            "Greedy approach selects the highest-probability token at each step (deterministic). "
            "Temperature and Top-K parameters are disabled."
        )
        temperature = 0.0
        top_k = None

    max_new_tokens = st.slider(
        label="Max New Tokens",
        min_value=10,
        max_value=300,
        value=100,
        step=10,
        help="Maximum number of tokens to generate beyond the prompt.",
    )

    use_seed = st.checkbox("Set Random Seed", value=False)
    if use_seed:
        seed_value = st.number_input("Seed Value", min_value=0, max_value=999999, value=42, step=1)
    else:
        seed_value = None

    st.markdown("---")
    st.subheader("Model Specifications")
    st.markdown(
        f"""
        - **Architecture:** Decoder-Only Transformer
        - **Parameters:** 50,221,056 (50.2M)
        - **Layers:** {MODEL_CONFIG['n_layers']}
        - **Heads:** {MODEL_CONFIG['n_heads']}
        - **Embedding Dim:** {MODEL_CONFIG['emb_dim']}
        - **Context Window:** {MODEL_CONFIG['context_length']} tokens
        - **Vocabulary:** {MODEL_CONFIG['vocab_size']} (GPT-2 BPE)
        - **Execution Device:** {device.upper()}
        """
    )


# Main Interface
st.markdown('<div class="main-title">LLM From Scratch: Text Generation</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">A custom decoder-only generative transformer implemented from first principles and pretrained on text corpora.</div>',
    unsafe_allow_html=True,
)

if not model_loaded:
    st.error(f"Error loading model checkpoint: {load_error_msg}")
    st.stop()

# Preset prompt selection
sample_prompts = [
    "Custom Prompt",
    "First Citizen:",
    "ROMEO:",
    "To be, or not to be:",
    "GLOUCESTER:",
    "KING LEAR:",
    "MENENIUS:",
]

selected_preset = st.selectbox("Preset Prompts", options=sample_prompts, index=1)

if selected_preset != "Custom Prompt":
    default_prompt_text = selected_preset
else:
    default_prompt_text = "First Citizen:"

prompt_input = st.text_area(
    label="Input Prompt",
    value=default_prompt_text,
    height=100,
    help="Enter the initial sequence of text to prompt the language model.",
)

col_actions = st.columns([1, 1, 6])
with col_actions[0]:
    generate_btn = st.button("Generate", type="primary", use_container_width=True)
with col_actions[1]:
    clear_btn = st.button("Clear", use_container_width=True)

if clear_btn:
    st.session_state.pop("generated_text", None)
    st.session_state.pop("generation_metrics", None)
    st.rerun()

# Execution Container
output_placeholder = st.empty()

if generate_btn:
    if not prompt_input.strip():
        st.warning("Please provide a prompt before generating.")
    else:
        if use_seed and seed_value is not None:
            torch.manual_seed(seed_value)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed_value)

        # Encode prompt
        encoded_ids = tokenizer.encode(prompt_input, allowed_special={"<|endoftext|>"})
        encoded_tensor = torch.tensor(encoded_ids).unsqueeze(0)
        prompt_token_count = len(encoded_ids)

        # Validate prompt length against context window
        if prompt_token_count >= MODEL_CONFIG["context_length"]:
            st.warning(
                f"Prompt length ({prompt_token_count} tokens) exceeds or matches context window "
                f"({MODEL_CONFIG['context_length']} tokens). The prompt will be truncated to fit."
            )

        start_time = time.perf_counter()

        # Stream output
        generated_token_ids = []
        full_text = prompt_input
        output_display = st.empty()

        for token_str, token_id in stream_generation(
            model=model,
            encoded_prompt=encoded_tensor,
            max_new_tokens=max_new_tokens,
            context_size=MODEL_CONFIG["context_length"],
            temperature=temperature,
            top_k=top_k,
            tokenizer=tokenizer,
            device=device,
        ):
            full_text += token_str
            generated_token_ids.append(token_id)
            output_display.text_area(
                label="Generated Output",
                value=full_text,
                height=220,
                disabled=True,
            )

        elapsed_time = time.perf_counter() - start_time
        gen_token_count = len(generated_token_ids)
        tokens_per_sec = gen_token_count / elapsed_time if elapsed_time > 0 else 0.0

        st.session_state["generated_text"] = full_text
        st.session_state["generation_metrics"] = {
            "prompt_tokens": prompt_token_count,
            "generated_tokens": gen_token_count,
            "total_tokens": prompt_token_count + gen_token_count,
            "time_seconds": elapsed_time,
            "tokens_per_sec": tokens_per_sec,
            "strategy": strategy,
            "temperature": temperature,
            "top_k": top_k,
            "generated_token_ids": generated_token_ids,
            "prompt_token_ids": encoded_ids,
        }

# Display cached result if available
if "generation_metrics" in st.session_state and not generate_btn:
    metrics = st.session_state["generation_metrics"]
    st.text_area(
        label="Generated Output",
        value=st.session_state["generated_text"],
        height=220,
        disabled=True,
    )

# Metrics display
if "generation_metrics" in st.session_state:
    metrics = st.session_state["generation_metrics"]

    st.markdown("### Performance and Metrics")
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    with m_col1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{metrics["prompt_tokens"]}</div><div class="metric-label">Prompt Tokens</div></div>',
            unsafe_allow_html=True,
        )
    with m_col2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{metrics["generated_tokens"]}</div><div class="metric-label">Generated Tokens</div></div>',
            unsafe_allow_html=True,
        )
    with m_col3:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{metrics["total_tokens"]}</div><div class="metric-label">Total Tokens</div></div>',
            unsafe_allow_html=True,
        )
    with m_col4:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{metrics["time_seconds"]:.2f}s</div><div class="metric-label">Latency</div></div>',
            unsafe_allow_html=True,
        )
    with m_col5:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{metrics["tokens_per_sec"]:.1f}</div><div class="metric-label">Tokens / Sec</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Detailed inspection tabs
    tab_text, tab_tokens, tab_arch = st.tabs(["Formatted Output", "Token Breakdown", "Architecture Details"])

    with tab_text:
        st.code(st.session_state["generated_text"], language="text")

    with tab_tokens:
        st.markdown("#### Prompt Token IDs")
        st.write(metrics["prompt_token_ids"])

        st.markdown("#### Generated Token IDs")
        st.write(metrics["generated_token_ids"])

    with tab_arch:
        st.markdown(
            """
            | Component | Configuration | Description |
            | :--- | :--- | :--- |
            | Vocabulary Size | 50,257 | Byte-Pair Encoding (BPE) tokenizer |
            | Context Length | 256 | Maximum sequence length supported |
            | Embedding Dimension | 384 | Hidden representation dimension |
            | Attention Heads | 6 | Multi-head self-attention heads |
            | Head Dimension | 64 | Dimension per attention head (384 / 6) |
            | Transformer Layers | 6 | Number of stacked transformer blocks |
            | Feed-Forward Expansion | 4x (1536) | Intermediate dimension with GELU |
            | Normalization | LayerNorm | eps = 1e-5 |
            | Causal Mask | Upper Triangular | Prevents attending to future tokens |
            """
        )
