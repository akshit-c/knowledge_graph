import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

# Set these in your environment (or defaults will work if paths match)
MLX_PYTHON = os.getenv("MLX_PYTHON", "")  # e.g. /Users/amit/Developer/mindlayer-training/venv/bin/python
BASE_MODEL = os.getenv("MLX_BASE_MODEL", "microsoft/Phi-3.5-mini-instruct")
ADAPTER_PATH = os.getenv("MLX_ADAPTER_PATH", "/Users/amit/Developer/mindlayer-training/outputs/mindlayer_core_v1")

def _python_exec() -> str:
    """
    Always prefer explicit MLX python to avoid conda/base python issues.
    """
    if MLX_PYTHON and os.path.exists(MLX_PYTHON):
        return MLX_PYTHON

    # If adapter path points into a training repo with its own venv,
    # prefer that Python (it should have MLX installed with matching arch).
    try:
        adapter = Path(ADAPTER_PATH).resolve()
        # e.g. /Users/amit/Developer/mindlayer-training/outputs/mindlayer_core_v1
        training_root = adapter.parents[1]  # .../mindlayer-training
        training_venv_python = training_root / "venv" / "bin" / "python"
        if training_venv_python.exists():
            return str(training_venv_python)
    except Exception:
        # Fallback to project/local env logic below
        pass

    # Try to find .venv/bin/python in the project root
    project_root = Path(__file__).parent.parent.parent
    venv_python = project_root / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    
    # Use the current Python interpreter (should be from venv if backend started correctly)
    return sys.executable

def clean_llm_text(txt: str) -> str:
    """Clean LLM output by removing special tokens."""
    if not txt:
        return ""
    for bad in ["<|end|>", "</s>", "<|eot_id|>", "<|endoftext|>"]:
        txt = txt.replace(bad, "")
    return txt.strip()

def clean_text(t: str) -> str:
    """Legacy function - use clean_llm_text instead."""
    return clean_llm_text(t)

def generate(prompt: str, max_tokens: int = 256) -> str:
    """
    Calls MLX-LM generate via python -m mlx_lm generate
    (works even if mlx_lm CLI isn't on PATH).
    """
    py = _python_exec()

    cmd = [
        py, "-m", "mlx_lm", "generate",
        "--model", BASE_MODEL,
        "--adapter-path", ADAPTER_PATH,
        "--prompt", prompt,
        "--max-tokens", str(max_tokens),
    ]

    p = subprocess.run(cmd, capture_output=True, text=True)

    if p.returncode != 0:
        raise RuntimeError(
            f"MLX generate failed\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}"
        )

    out = p.stdout.strip()
    lines = [ln.rstrip() for ln in out.splitlines()]

    # MLX prints:
    # ==========
    # <ANSWER>
    # ==========
    # Prompt: ...
    # Generation: ...
    # Peak memory: ...
    #
    # We'll extract the first block between "==========" markers.
    marker = "=========="
    idxs = [i for i, ln in enumerate(lines) if ln.strip() == marker]

    if len(idxs) >= 2:
        start, end = idxs[0], idxs[1]
        answer_lines = [ln.strip() for ln in lines[start + 1 : end] if ln.strip()]
        answer = "\n".join(answer_lines).strip()
        return clean_llm_text(answer)

    # Fallback: remove known non-answer lines and return remaining last line
    junk_prefixes = ("Prompt:", "Generation:", "Peak memory:", "Fetching", "Download complete")
    filtered = []
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        if any(s.startswith(j) for j in junk_prefixes):
            continue
        filtered.append(s)

    answer = filtered[-1] if filtered else ""
    return clean_llm_text(answer)
