import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Configuration
MODEL_NAME = "google/gemma-2b-it"  # Smaller version for local testing
DEFAULT_MAX_LENGTH = 512

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for settings
with st.sidebar:
    st.header("Configuration")
    hf_token = st.text_input("Hugging Face Token", type="password", help="Required for Gemma access")
    max_length = st.slider("Max Response Length", 100, 1024, DEFAULT_MAX_LENGTH)
    device_map = "cuda" if torch.cuda.is_available() else "cpu"
    st.write(f"Running on: {device_map.upper()}")

# Model loading function
@st.cache_resource
def load_model(hf_token):
    if not hf_token:
        st.error("Hugging Face Token required!")
        return None, None
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            token=hf_token,
            torch_dtype=torch.bfloat16,
            device_map=device_map
        )
        return model, tokenizer
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None, None

# Load model
model, tokenizer = load_model(hf_token)

# Main UI
st.title("🧠 Local Gemma Assistant")
mode = st.selectbox("Select Mode", [
    "Chat",
    "Text Generation",
    "Code Completion",
    "Summarization"
])

# Generation function
def generate_response(prompt, mode):
    try:
        if mode == "Chat":
            chat_template = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": ""}
            ]
            inputs = tokenizer.apply_chat_template(chat_template, return_tensors="pt").to(model.device)
        else:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        outputs = model.generate(
            inputs,
            max_new_tokens=max_length,
            temperature=0.7,
            do_sample=True
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        return f"Error: {str(e)}"

# Mode handling
if mode == "Chat":
    st.header("Chat Mode")
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Your message"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.spinner("Thinking..."):
            response = generate_response(prompt, mode)
            st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

elif mode == "Code Completion":
    st.header("Code Assistant")
    code_prompt = st.text_area("Enter code problem:", height=150)
    if st.button("Generate Code"):
        with st.spinner("Coding..."):
            response = generate_response(f"Write Python code for: {code_prompt}", mode)
            st.code(response, language="python")

elif mode == "Summarization":
    st.header("Summarization Mode")
    text = st.text_area("Paste your text:", height=300)
    if st.button("Summarize"):
        with st.spinner("Processing..."):
            response = generate_response(f"Summarize this text:\n{text}", mode)
            st.write(response)

# Run with: streamlit run app.py