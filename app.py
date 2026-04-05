import streamlit as st
import torch
import matplotlib.pyplot as plt
import numpy as np
from transformers import BertTokenizer
from config import Config
from models.bidaf import BiDAF
from models.bert_bidaf import BertBiDAF
from data.squad_loader import SQuADBiDAFDataset
from utils.metrics import normalize_answer

cfg = Config()

@st.cache_resource
def load_bidaf_model(checkpoint_path, word2idx, char2idx):
    model = BiDAF(len(word2idx), len(char2idx))
    model.load_state_dict(torch.load(checkpoint_path, map_location=cfg.device))
    model.to(cfg.device)
    model.eval()
    return model

@st.cache_resource
def load_bert_model(checkpoint_path):
    model = BertBiDAF()
    model.load_state_dict(torch.load(checkpoint_path, map_location=cfg.device))
    model.to(cfg.device)
    model.eval()
    return model

def tokenize_for_bidaf(context, question, word2idx, char2idx):
    """Tokenize context and question for BiDAF (without padding)."""
    def word_to_ids(words):
        return [word2idx.get(w.lower(), 1) for w in words]
    def char_to_ids(word):
        return [char2idx.get(ch, 1) for ch in word[:cfg.max_word_len]]

    c_words = context.split()
    q_words = question.split()
    c_word_ids = word_to_ids(c_words)
    q_word_ids = word_to_ids(q_words)
    c_char_ids = [char_to_ids(w) for w in c_words]
    q_char_ids = [char_to_ids(w) for w in q_words]
    # Pad char sequences to max word length in this example
    max_clen = max(len(w) for w in c_char_ids) if c_char_ids else 0
    max_qlen = max(len(w) for w in q_char_ids) if q_char_ids else 0
    max_char_len = max(max_clen, max_qlen, 1)
    c_char_ids = [c + [0]*(max_char_len - len(c)) for c in c_char_ids]
    q_char_ids = [c + [0]*(max_char_len - len(c)) for c in q_char_ids]
    # Convert to tensors and add batch dimension
    c_word_t = torch.tensor([c_word_ids])
    c_char_t = torch.tensor([c_char_ids])
    q_word_t = torch.tensor([q_word_ids])
    q_char_t = torch.tensor([q_char_ids])
    return c_word_t, c_char_t, q_word_t, q_char_t, c_words

def predict_bidaf(model, c_word, c_char, q_word, q_char):
    with torch.no_grad():
        start_logits, end_logits = model(c_word, c_char, q_word, q_char)
    start_probs = torch.softmax(start_logits, dim=1)[0].cpu().numpy()
    end_probs   = torch.softmax(end_logits, dim=1)[0].cpu().numpy()
    start_idx = np.argmax(start_probs)
    end_idx   = np.argmax(end_probs)
    return start_idx, end_idx, start_probs, end_probs

def predict_bert(model, question, context):
    tokenizer = BertTokenizer.from_pretrained(cfg.bert_name, model_max_length=cfg.max_seq_len)
    inputs = tokenizer(
        question, context,
        max_length=cfg.max_seq_len,
        truncation='only_second',
        padding='max_length',
        return_tensors='pt'
    )
    input_ids = inputs['input_ids'].to(cfg.device)
    attn_mask = inputs['attention_mask'].to(cfg.device)
    token_type_ids = inputs['token_type_ids'].to(cfg.device)
    with torch.no_grad():
        start_logits, end_logits = model(input_ids, attn_mask, token_type_ids)
    start_probs = torch.softmax(start_logits, dim=1)[0].cpu().numpy()
    end_probs   = torch.softmax(end_logits, dim=1)[0].cpu().numpy()
    start_idx = np.argmax(start_probs)
    end_idx   = np.argmax(end_probs)
    # Decode answer tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    answer_tokens = tokens[start_idx:end_idx+1]
    answer = tokenizer.convert_tokens_to_string(answer_tokens)
    return answer, start_idx, end_idx, start_probs, end_probs, tokens

def plot_probs(probs, tokens, title):
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.bar(range(len(probs)), probs, color='skyblue')
    ax.set_xticks(range(len(probs)))
    ax.set_xticklabels(tokens, rotation=90, fontsize=8)
    ax.set_title(title)
    ax.set_ylabel('Probability')
    st.pyplot(fig)

def main():
    st.set_page_config(page_title="BiDAF / BERT+BiDAF QA Visualizer", layout="wide")
    st.title("📖 Question Answering with BiDAF & BERT+BiDAF")
    st.markdown("Enter a context passage and a question to see the model's predicted answer and token‑level probabilities.")

    # Sidebar – model selection and checkpoint loading
    st.sidebar.header("Model Configuration")
    model_type = st.sidebar.selectbox("Model Type", ["BiDAF (pure)", "BERT+BiDAF"])

    if model_type == "BiDAF (pure)":
        # For pure BiDAF we need vocab from a dataset instance
        # Load a dummy dataset to get vocab (cached)
        @st.cache_resource
        def get_vocab():
            dataset = SQuADBiDAFDataset(split='train', use_bert=False)
            return dataset.word2idx, dataset.char2idx
        word2idx, char2idx = get_vocab()
        checkpoint = st.sidebar.file_uploader("Load BiDAF checkpoint (.pt)", type=['pt'])
        if checkpoint is not None:
            # Save uploaded file temporarily
            with open("temp_bidaf.pt", "wb") as f:
                f.write(checkpoint.getbuffer())
            model = load_bidaf_model("temp_bidaf.pt", word2idx, char2idx)
            st.sidebar.success("BiDAF model loaded")
        else:
            st.sidebar.warning("Please upload a BiDAF checkpoint")
            model = None
    else:
        checkpoint = st.sidebar.file_uploader("Load BERT+BiDAF checkpoint (.pt)", type=['pt'])
        if checkpoint is not None:
            with open("temp_bert.pt", "wb") as f:
                f.write(checkpoint.getbuffer())
            model = load_bert_model("temp_bert.pt")
            st.sidebar.success("BERT+BiDAF model loaded")
        else:
            st.sidebar.warning("Please upload a BERT+BiDAF checkpoint")
            model = None

    # Main input area
    col1, col2 = st.columns(2)
    with col1:
        context = st.text_area("📄 Context", "The Eiffel Tower is located in Paris, France. It was built in 1889 and is one of the most famous landmarks in the world.", height=200)
    with col2:
        question = st.text_input("❓ Question", "Where is the Eiffel Tower located?")

    if st.button("Predict Answer") and model is not None and context and question:
        st.subheader("Prediction Results")

        if model_type == "BiDAF (pure)":
            c_word, c_char, q_word, q_char, context_tokens = tokenize_for_bidaf(context, question, word2idx, char2idx)
            c_word, c_char, q_word, q_char = [x.to(cfg.device) for x in [c_word, c_char, q_word, q_char]]
            start_idx, end_idx, start_probs, end_probs = predict_bidaf(model, c_word, c_char, q_word, q_char)
            answer_tokens = context_tokens[start_idx:end_idx+1]
            answer = " ".join(answer_tokens)
            st.success(f"**Answer:** {answer}")
            st.info(f"Span: tokens {start_idx} – {end_idx}  |  Confidence (start): {start_probs[start_idx]:.3f}, (end): {end_probs[end_idx]:.3f}")
            # Show probability plots
            plot_probs(start_probs, context_tokens, "Start Token Probabilities")
            plot_probs(end_probs, context_tokens, "End Token Probabilities")
        else:
            answer, start_idx, end_idx, start_probs, end_probs, tokens = predict_bert(model, question, context)
            st.success(f"**Answer:** {answer}")
            st.info(f"Token indices: {start_idx} – {end_idx}  |  Confidence (start): {start_probs[start_idx]:.3f}, (end): {end_probs[end_idx]:.3f}")
            # Show probability plots (only up to max_seq_len tokens, but we can show all non‑padding)
            valid_len = len([t for t in tokens if t not in ['[PAD]', '[CLS]', '[SEP]']])
            plot_probs(start_probs[:valid_len], tokens[:valid_len], "Start Token Probabilities (BERT)")
            plot_probs(end_probs[:valid_len], tokens[:valid_len], "End Token Probabilities (BERT)")

        # Optionally show attention from BERT (requires modifying BertBiDAF to return attentions)
        if model_type == "BERT+BiDAF" and hasattr(model.bert, 'encoder'):
            st.subheader("BERT Attention Visualization (Last Layer)")
            st.markdown("*This requires modifying `BertBiDAF.forward` to return `outputs.attentions`.*")

    st.markdown("---")
    st.caption("Visualization shows the probability distribution over token positions for the start and end of the answer span.")

if __name__ == "__main__":
    main()