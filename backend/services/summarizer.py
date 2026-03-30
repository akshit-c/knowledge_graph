from backend.services.local_llm_mlx import generate

def build_summary_prompt(text: str) -> str:
    return f"""Summarize the following document in 4 to 6 bullet points.
Rules:
- Only factual points from the text
- No extra commentary
- If the text is too short or unclear, output exactly: NOT_IN_MEMORY

TEXT:
{text}
"""

def summarize_document(text: str, max_chars: int = 6000) -> str:
    # Keep it small for speed + stability
    sliced = (text or "").strip()[:max_chars]
    if not sliced:
        return ""
    out = generate(build_summary_prompt(sliced)).strip()
    # clean any leftovers
    for bad in ["<|end|>", "</s>", "<|eot_id|>"]:
        out = out.replace(bad, "")
    return out.strip()
