import re

def sanitize_input(text, max_length=2000):
    cleaned = re.sub(r"[^\w\s.,!?\'\"-]", "", text)
    return cleaned.strip()[:max_length]