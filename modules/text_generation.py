import streamlit as st

class TextGenerationUI:
    def __init__(self, agent):
        self.agent = agent
        self.modes = {
            "Creative Writing": {
                "styles": ["Magical Realism", "Cyberpunk", "Historical Fiction", "Mystery"],
                "examples": ["A library where books rewrite themselves", "A heist on a floating city"]
            },
            "Poetry": {
                "forms": ["Sonnet", "Haiku", "Free Verse", "Limerick"],
                "themes": ["Love", "Technology", "Nature", "Time"]
            },
            "Summarization": {
                "focus_areas": ["Technical Details", "Key Arguments", "Main Conclusions", "Methodology"]
            }
        }
    
    def render_controls(self):
        col1, col2 = st.columns([3, 1])
        with col1:
            sub_mode = st.selectbox("Select Sub-Mode", list(self.modes.keys()))
        with col2:
            temperature = st.slider("Creativity", 0.5, 1.5, 1.0, 0.1)
        
        return sub_mode, temperature
    
    def render_inputs(self, sub_mode):
        prompt = ""
        
        if sub_mode == "Creative Writing":
            col1, col2 = st.columns(2)
            with col1:
                style = st.selectbox("Writing Style", self.modes[sub_mode]["styles"])
            with col2:
                st.caption("Example Prompts")
                example = st.selectbox("Choose example", self.modes[sub_mode]["examples"])
            prompt = st.text_area("Your Writing Prompt", value=example)
            full_prompt = self.agent.system_prompts["text_gen"]["creative_writing"].format(
                style=style,
                prompt=prompt
            )
        
        elif sub_mode == "Poetry":
            col1, col2, col3 = st.columns([2,2,1])
            with col1:
                form = st.selectbox("Poetic Form", self.modes[sub_mode]["forms"])
            with col2:
                theme = st.selectbox("Theme", self.modes[sub_mode]["themes"])
            with col3:
                lines = st.number_input("Lines", 3, 20, 12)
            tone = st.selectbox("Tone", ["Melancholic", "Joyful", "Reflective", "Satirical"])
            prompt = self.agent.system_prompts["text_gen"]["poetry"].format(
                form=form,
                theme=theme,
                lines=lines,
                tone=tone
            )
        
        elif sub_mode == "Summarization":
            text = st.text_area("Paste your text", height=300)
            col1, col2 = st.columns(2)
            with col1:
                length = st.selectbox("Summary Length", ["Concise (100 words)", "Detailed (300 words)"])
            with col2:
                focus = st.multiselect("Focus Areas", self.modes[sub_mode]["focus_areas"])
            prompt = self.agent.system_prompts["text_gen"]["summarization"].format(
                text=text,
                length=length.split(" ")[1][1:],
                focus_areas=", ".join(focus)
            )
        
        return prompt