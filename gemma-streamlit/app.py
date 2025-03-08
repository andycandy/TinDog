import streamlit as st
from modules.base_agent import GemmaAgent
from modules.text_generation import TextGenerationUI
from utils.helpers import sanitize_input

if "agent" not in st.session_state:
    st.session_state.agent = None

with st.sidebar:
    st.header("🔧 Configuration")
    hf_token = st.text_input("HF Token", type="password")
    if hf_token and not st.session_state.agent:
        try:
            st.session_state.agent = GemmaAgent(hf_token)
            st.success("Model loaded!")
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
    
    st.divider()
    st.caption("Current Mode: Text Generation")

# Main interface
st.title("✍️ Gemma Text Studio")

if st.session_state.agent:
    text_ui = TextGenerationUI(st.session_state.agent)
    
    # Render controls
    sub_mode, temperature = text_ui.render_controls()
    
    # Get formatted prompt
    prompt = text_ui.render_inputs(sub_mode)
    
    if st.button("Generate", type="primary"):
        clean_prompt = sanitize_input(prompt)
        with st.spinner("Crafting your text..."):
            response = st.session_state.agent.generate(
                clean_prompt,
                temperature=temperature,
                max_length=1024
            )
        
        st.subheader("Generated Result")
        if sub_mode == "Poetry":
            st.write(response.replace("\n", "  \n"))  # Preserve line breaks
        elif sub_mode == "Creative Writing":
            st.write(response)
        else:
            st.markdown(response)
else:
    st.warning("Please enter your Hugging Face token in the sidebar")