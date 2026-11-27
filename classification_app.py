import os
import sys
import time
from pathlib import Path
import tiktoken
import torch
import streamlit as st

# Add src to python path for module imports
src_dir = Path(__file__).resolve().parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from llmfromscratch.LLMArchitecture import LLM
from llmfromscratch.LoRA import LinearWithLoRA

# Default checkpoint path
CHECKPOINT_PATH = Path(__file__).resolve().parent / "models" / "classification_finetuned_model.pth"
MAX_LENGTH = 128
NUM_CLASSES = 2
LORA_RANK = 16
LORA_ALPHA = 16
PAD_TOKEN_ID = 50256
LABEL_NAMES = {0: "Negative", 1: "Positive"}

CONFIG = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 384,
    "n_heads": 6,
    "n_layers": 6,
    "drop_rate": 0.2,
    "qkv_bias": False,
}

st.set_page_config(
    page_title="Sentiment Classifier - Fine-Tuned LLM",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (professional, zero emojis)
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.0rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 0.95rem;
        color: #6c757d;
        margin-bottom: 1.5rem;
    }
    .result-card-positive {
        background-color: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 1.2rem;
    }
    .result-card-negative {
        background-color: #fef2f2;
        border: 1px solid #fecaca;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 1.2rem;
    }
    .badge-positive {
        display: inline-block;
        padding: 4px 12px;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        border-radius: 4px;
        background-color: #16a34a;
        color: #ffffff;
    }
    .badge-negative {
        display: inline-block;
        padding: 4px 12px;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        border-radius: 4px;
        background-color: #dc2626;
        color: #ffffff;
    }
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 6px;
        padding: 12px 16px;
        text-align: center;
    }
    .metric-value {
        font-size: 1.25rem;
        font-weight: 600;
        color: #212529;
    }
    .metric-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #6c757d;
        margin-top: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def replace_linear_with_lora(model, rank, alpha):
    """Replaces linear layers with LoRA adapters except multi-head attention."""
    for name, module in model.named_children():
        if isinstance(module, torch.nn.MultiheadAttention):
            continue
        if isinstance(module, torch.nn.Linear):
            setattr(model, name, LinearWithLoRA(module, rank, alpha))
        else:
            replace_linear_with_lora(module, rank, alpha)


def last_token_logits(model, input_batch, pad_id=PAD_TOKEN_ID):
    """Extracts logits at the final non-padded token position in the sequence."""
    logits = model(input_batch)
    lengths = (input_batch != pad_id).sum(dim=1)
    idx = (lengths - 1).clamp(min=0)
    return logits[torch.arange(logits.size(0), device=logits.device), idx, :], idx


@st.cache_resource(show_spinner=False)
def load_classifier(checkpoint_path: Path):
    """Initializes model, attaches LoRA adapters and linear classification head, then loads weights."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    model = LLM(CONFIG)
    model.out_head = torch.nn.Linear(CONFIG["emb_dim"], NUM_CLASSES)
    replace_linear_with_lora(model, rank=LORA_RANK, alpha=LORA_ALPHA)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    tokenizer = tiktoken.get_encoding("gpt2")
    return model, tokenizer, device


def evaluate_text(text: str, model, tokenizer, device: str, max_length: int = MAX_LENGTH):
    """Runs tokenization, inference, and softmax probability extraction."""
    tokens = tokenizer.encode(text)[:max_length]
    if len(tokens) == 0:
        raise ValueError("Provided text is empty after tokenization.")

    token_count = len(tokens)
    padded = tokens + [PAD_TOKEN_ID] * (max_length - len(tokens))
    input_tensor = torch.tensor(padded, device=device).unsqueeze(0)

    with torch.no_grad():
        logits, last_idx = last_token_logits(model, input_tensor, PAD_TOKEN_ID)

    probs = torch.softmax(logits, dim=-1)[0]
    predicted_idx = int(torch.argmax(probs).item())

    return {
        "label": LABEL_NAMES[predicted_idx],
        "class_id": predicted_idx,
        "confidence": probs[predicted_idx].item(),
        "prob_negative": probs[0].item(),
        "prob_positive": probs[1].item(),
        "logits_negative": logits[0, 0].item(),
        "logits_positive": logits[0, 1].item(),
        "token_count": token_count,
        "token_ids": tokens,
        "evaluated_token_index": last_idx.item(),
    }


# Load model
try:
    model, tokenizer, device = load_classifier(CHECKPOINT_PATH)
    model_loaded = True
except Exception as exc:
    model_loaded = False
    load_error_msg = str(exc)

# Sidebar
with st.sidebar:
    st.subheader("Model Specifications")
    st.markdown(
        f"""
        - **Architecture:** 6-Layer Decoder Transformer
        - **Base Parameters:** 50,221,056 (~50.2M)
        - **Fine-Tuning:** Low-Rank Adaptation (LoRA)
        - **LoRA Rank ($r$):** {LORA_RANK}
        - **LoRA Alpha ($\alpha$):** {LORA_ALPHA}
        - **Output Classes:** 2 (Negative, Positive)
        - **Context Window:** {MAX_LENGTH} tokens
        - **Padding Token:** {PAD_TOKEN_ID} (`<|endoftext|>`)
        - **Inference Device:** {device.upper()}
        """
    )

    st.markdown("---")
    st.subheader("Configuration")
    selected_max_len = st.slider(
        label="Context Window Limit",
        min_value=32,
        max_value=128,
        value=MAX_LENGTH,
        step=16,
        help="Controls the truncation threshold before padding to evaluate the sequence.",
    )

    st.markdown("---")
    st.caption(
        "The model uses causal self-attention pooling: the representation at the final non-padded token "
        "aggregates full left-to-right sentence context and is projected by the classification head."
    )


# Main Interface Header
st.markdown('<div class="main-title">LLM Fine-Tuned Sentiment Classifier</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">A clean demonstration of parameter-efficient classification fine-tuning using LoRA on a custom transformer.</div>',
    unsafe_allow_html=True,
)

if not model_loaded:
    st.error(f"Error loading model checkpoint: {load_error_msg}")
    st.stop()

# Presets selection
presets = {
    "Select an example or enter custom text below": "",
    "Positive: Joy and Greeting (User Sample)": "O, what a joy it is to see thee well, my dearest friend!",
    "Negative: Accusation and Curse (User Sample)": "Thou art a villain, and I curse the day that gave thee birth.",
    "Neutral/Mild: Royal Movement (User Sample)": "The king hath gone to the city and will return by noon.",
    "Positive: Grateful Arrival (The Winter's Tale)": (
        "Most royal sir, from thence; from him, whose daughter His tears proclaim'd his, parting with her: thence, "
        "A prosperous south-wind friendly, we have cross'd, To execute the charge my father gave me For visiting your highness: "
        "my best train I have from your Sicilian shores dismiss'd; Who for Bohemia bend, to signify Not only my success in Libya, sir, "
        "But my arrival and my wife's in safety Here where we are."
    ),
    "Negative: Grievance and Slaughter (Richard III)": (
        "That thou hast wronged in the time o'erpast; For I myself have many tears to wash Hereafter time, for time past wrong'd by thee. "
        "The children live, whose parents thou hast slaughter'd, Ungovern'd youth, to wail it in their age; "
        "The parents live, whose children thou hast butcher'd, Old wither'd plants, to wail it with their age. "
        "Swear not by time to come; for that thou hast Misused ere used, by time misused o'erpast."
    ),
    "Negative: Invective (King John)": "Thou slave, thou wretch, thou coward! Thou little valiant, great in villainy! Thou ever strong upon the stronger side!",
}

selected_preset = st.selectbox("Example Passages", options=list(presets.keys()), index=0)
default_input = presets[selected_preset] if presets[selected_preset] else ""

text_area_value = st.text_area(
    label="Input Passage",
    value=default_input,
    height=130,
    placeholder="Enter text to classify sentiment...",
)

col_actions = st.columns([1, 1, 6])
with col_actions[0]:
    classify_button = st.button("Classify Sentiment", type="primary", use_container_width=True)
with col_actions[1]:
    clear_button = st.button("Clear", use_container_width=True)

if clear_button:
    st.session_state.pop("eval_result", None)
    st.rerun()

if classify_button:
    if not text_area_value.strip():
        st.warning("Please enter text before running classification.")
    else:
        start_clock = time.perf_counter()
        try:
            eval_data = evaluate_text(
                text=text_area_value,
                model=model,
                tokenizer=tokenizer,
                device=device,
                max_length=selected_max_len,
            )
            eval_data["latency_ms"] = (time.perf_counter() - start_clock) * 1000
            st.session_state["eval_result"] = eval_data
        except Exception as err:
            st.error(f"Inference error: {str(err)}")

# Results Display
if "eval_result" in st.session_state:
    data = st.session_state["eval_result"]
    is_pos = data["label"] == "Positive"

    card_class = "result-card-positive" if is_pos else "result-card-negative"
    badge_class = "badge-positive" if is_pos else "badge-negative"

    st.markdown(
        f"""
        <div class="{card_class}">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span class="{badge_class}">{data['label'].upper()}</span>
                    <span style="font-size: 1.15rem; font-weight: 600; margin-left: 14px; color: #1e293b;">
                        Confidence: {data['confidence']:.1%}
                    </span>
                </div>
                <div style="font-size: 0.85rem; color: #64748b;">
                    Evaluated at token index {data['evaluated_token_index']} of {selected_max_len}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{data["label"]}</div><div class="metric-label">Predicted Class</div></div>',
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{data["confidence"]:.1%}</div><div class="metric-label">Confidence</div></div>',
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{data["token_count"]} / {selected_max_len}</div><div class="metric-label">Tokens Used</div></div>',
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{data["latency_ms"]:.1f} ms</div><div class="metric-label">Inference Latency</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.subheader("Probability Distribution")

    col_neg, col_pos = st.columns(2)
    with col_neg:
        st.write(f"Negative Sentiment: **{data['prob_negative']:.2%}**")
        st.progress(data["prob_negative"])
    with col_pos:
        st.write(f"Positive Sentiment: **{data['prob_positive']:.2%}**")
        st.progress(data["prob_positive"])

    st.markdown("---")

    # Detailed Inspection Tabs
    tab_logits, tab_tokens, tab_architecture = st.tabs(
        ["Logits and Probability Distribution", "Tokenized Representation", "LoRA Architecture Insights"]
    )

    with tab_logits:
        st.markdown("#### Raw Output Logits (Pre-Softmax)")
        st.markdown(
            f"""
            | Class | Raw Logit ($z_i$) | Softmax Probability ($P_i$) |
            | :--- | :--- | :--- |
            | **Negative (0)** | `{data['logits_negative']:.4f}` | `{data['prob_negative']:.4f}` ({data['prob_negative']:.1%}) |
            | **Positive (1)** | `{data['logits_positive']:.4f}` | `{data['prob_positive']:.4f}` ({data['prob_positive']:.1%}) |
            """
        )
        st.caption(
            f"Logits are retrieved from the final non-padded position: index {data['evaluated_token_index']}."
        )

    with tab_tokens:
        st.markdown(f"#### Sequence Breakdown ({data['token_count']} active tokens)")
        token_strings = [tokenizer.decode([tid]) for tid in data["token_ids"]]
        table_rows = [
            {"Index": i, "Subword Token": repr(t_str), "Token ID": t_id}
            for i, (t_str, t_id) in enumerate(zip(token_strings, data["token_ids"]))
        ]
        st.dataframe(table_rows, use_container_width=True, height=230)
        st.caption(
            f"The sequence is right-padded with {selected_max_len - data['token_count']} padding tokens "
            f"(ID: {PAD_TOKEN_ID}) to form a uniform tensor of length {selected_max_len}."
        )

    with tab_architecture:
        st.markdown("#### Parameter-Efficient Fine-Tuning with LoRA")
        st.markdown(
            """
            Instead of updating the full 50.2M base model parameters, Low-Rank Adaptation freezes 
            the base pre-trained weights and adds low-rank decomposition matrices to linear projections:

            $$W = W_0 + \\Delta W = W_0 + \\frac{\\alpha}{r} (A \\times B)$$

            Where:
            - $W_0 \\in \\mathbb{R}^{d_{\\text{in}} \\times d_{\\text{out}}}$: Frozen pre-trained matrix.
            - $A \\in \\mathbb{R}^{d_{\\text{in}} \\times r}$: Initialized with Kaiming uniform distribution.
            - $B \\in \\mathbb{R}^{r \\times d_{\\text{out}}}$: Initialized with zeros so training starts at $W_0$.
            - Rank ($r$): 16
            - Scaling factor ($\\alpha$): 16

            **Why Last-Token Classification?**
            In autoregressive decoder transformers, attention is causal (masked upper-triangular). 
            Each token can only attend to previous tokens. Therefore, the final non-padded token has 
            accumulated contextual attention from all preceding tokens, providing an optimal summary 
            representation for sequence-level classification.
            """
        )
