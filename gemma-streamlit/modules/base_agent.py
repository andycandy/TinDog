import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class GemmaAgent:
    def __init__(self, hf_token, model_name="google/gemma-2b-it"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=hf_token,
            torch_dtype=torch.bfloat16,
            device_map=self.device
        )
        self.system_prompts = {}
        self._load_text_generation_prompts()
    
    def _load_text_generation_prompts(self):
        self.system_prompts["text_gen"] = {
            "creative_writing": (
                "You are an award-winning author. Generate vivid, imaginative text in the style of {style}. "
                "Focus on sensory details and emotional resonance. Respond only with the text, no explanations.\n\n"
                "Prompt: {prompt}"
            ),
            "poetry": (
                "You are a poet laureate. Create a {form} poem that explores the theme of {theme}. "
                "Use vivid imagery and rhythmic language. Include at least {lines} lines.\n\n"
                "Additional instructions: {tone} tone"
            ),
            "summarization": (
                "You are a research assistant. Summarize this text while preserving key information and technical terms. "
                "Target length: {length} words. Focus on: {focus_areas}.\n\n"
                "Text: {text}"
            )
        }
    
    def generate(self, prompt, temperature=1.0, max_length=512):
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096  # Gemma's context window
        ).to(self.device)
        
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)