import streamlit as st
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
from pathlib import Path
import os

# Set up the app
st.set_page_config(
    page_title="IMDB Fine-Tuned DistilBERT",
    page_icon="🎬",
    layout="wide"
)

# Define the path to your model files
MODEL_PATH = r"C:\Users\youss\OneDrive\Desktop\LLM Tasks"

@st.cache_resource
def load_components():
    """Load model and tokenizer with error handling"""
    try:
        # Verify files exist
        required_files = [
            "config.json",
            "model.safetensors",  # or pytorch_model.bin for older versions
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.txt"
        ]
        
        # Check if each required file exists
        missing_files = [f for f in required_files if not os.path.exists(os.path.join(MODEL_PATH, f))]
        if missing_files:
            st.error(f"Missing model files: {', '.join(missing_files)}")
            return None, None
        
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForMaskedLM.from_pretrained(
            MODEL_PATH,
            local_files_only=True
        )
        return model, tokenizer
        
    except Exception as e:
        st.error(f"Loading failed: {str(e)}")
        return None, None

def predict_masks(text, model, tokenizer, top_k=5):
    """Generate predictions for masked tokens"""
    inputs = tokenizer(text, return_tensors="pt")
    mask_positions = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    predictions = []
    for mask_pos in mask_positions:
        logits = outputs.logits[0, mask_pos]
        probs = torch.nn.functional.softmax(logits, dim=-1)
        top_tokens = torch.topk(probs, top_k)
        
        for token_id, prob in zip(top_tokens.indices, top_tokens.values):
            token = tokenizer.decode(token_id)
            predictions.append({
                "token": token,
                "score": float(prob),
                "position": mask_pos.item()
            })
    
    return predictions

def main():
    st.title("🎬 IMDB Review Masked Word Prediction")
    st.markdown("Using our fine-tuned DistilBERT model trained on IMDB reviews")
    
    # Load components
    model, tokenizer = load_components()
    if model is None or tokenizer is None:
        st.error("Failed to initialize model. Please check the model files.")
        return
    
    # Input section
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("Input Text")
        default_example = "The movie plot was [MASK] and the acting was [MASK]!"
        user_input = st.text_area(
            "Enter text with [MASK] tokens:",
            value=default_example,
            height=150
        )
        
        top_k = st.slider("Predictions per mask:", 1, 10, 3)
        predict_btn = st.button("Generate Predictions", type="primary")

    # Results section
    with col2:
        st.subheader("Predictions")
        
        if predict_btn:
            if "[MASK]" not in user_input:
                st.warning("Please include [MASK] tokens in your text")
            else:
                with st.spinner("Analyzing..."):
                    predictions = predict_masks(user_input, model, tokenizer, top_k)
                
                if not predictions:
                    st.error("No [MASK] tokens found in input")
                    return
                
                # Display predictions
                current_pos = None
                tokens = user_input.split()
                
                for pred in predictions:
                    if pred["position"] != current_pos:
                        st.markdown(f"**Position {pred['position']} predictions:**")
                        current_pos = pred["position"]
                    
                    st.write(f"- {pred['token']} ({pred['score']:.1%})")
                
                # Show completed sentence
                completed = user_input
                for pred in sorted(predictions, key=lambda x: x["position"], reverse=True):
                    if pred["score"] > 0.1:  # Only replace if confident
                        completed = completed.replace(
                            "[MASK]", 
                            f"**{pred['token']}**", 
                            1
                        )
                
                st.divider()
                st.subheader("Completed Text")
                st.markdown(completed)

    # Examples section
    st.divider()
    st.subheader("Try These Examples")
    
    examples = [
        "The cinematography was [MASK] but the script felt [MASK].",
        "This [MASK] performance deserves an Oscar!",
        "I've never seen a more [MASK] ending in my life!",
        "The [MASK] director really [MASK] this story to life."
    ]
    
    cols = st.columns(len(examples))
    for col, ex in zip(cols, examples):
        if col.button(ex[:25] + "..." if len(ex) > 25 else ex):
            st.session_state.input_text = ex

if __name__ == "__main__":
    main()