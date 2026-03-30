import json
from pathlib import Path

POS = [
  "What is MindLayer in one sentence?",
  "What problem does MindLayer solve?",
  "What does “privacy-first” mean in the context of MindLayer?",
  "What makes MindLayer different from a normal chatbot?",
  "What is “grounded intelligence” in MindLayer?",
  "What is the ingestion pipeline in this system?",
  "Which file types can MindLayer ingest?",
  "Why do we chunk documents before indexing?",
  "What are semantic embeddings used for here?",
  "What is vector search and why is it used?",
  "What role does FAISS play in this project?",
  "What is stored in `memory.sqlite`?",
  "What is stored in `faiss.index` and `metadata.pkl`?",
  "What is the purpose of the “NOT_IN_MEMORY” behavior?",
  "How does the system prevent hallucinations?",
  "What does “top_k” control during retrieval?",
  "What is the difference between retrieval and generation?",
  "What is RAG in simple terms for this project?",
  "What is the benefit of separating ingestion, storage, retrieval, and reasoning?",
  "What does the “summary-first” mode do?",
  "When should summary-first mode be used?",
  "What is the role of document summaries in answering questions?",
  "What is the difference between full-context and summary-context answering?",
  "What does “kg-first” or “kg-only” mean in your trace output?",
  "What is the purpose of Neo4j in this architecture?",
  "What is a Claim node in the knowledge graph?",
  "What is an Entity node in the knowledge graph?",
  "What are MENTIONS edges used for?",
  "What are ABOUT edges used for?",
  "What does “support” mean in the KG claim ranking output?",
  "What is evidence in the KG-based answer output?",
  "How does your system link a claim to chunk evidence?",
  "What is the advantage of “KG claim + evidence” context?",
  "What kind of insights can MindLayer provide over time?",
  "What is the “Memory Timeline View” supposed to show?",
  "What is the “Writing Coach” feature meant to do?",
  "What is “Sentiment & Tone Analysis” used for?",
  "How can MindLayer help with decision-making?",
  "What is the “Smart Pinboard” idea?",
  "What does “cross-platform sync” imply in your design?",
  "What is “local-only privacy vault mode”?",
  "What is the purpose of metadata tagging?",
  "What kind of metadata fields are useful (examples)?",
  "What is the difference between storing PDFs and storing parsed text?",
  "What does “multimodal parsing” mean in your plan?",
  "Why is auditability important for this system?",
  "Why is reproducibility important for research evaluation?",
  "How does the system choose between NOT_IN_MEMORY and answering?",
  "What similarity threshold gating are you using (conceptually)?",
  "Why can weak retrieval cause hallucinations?",
  "What is the role of LoRA fine-tuning in your project?",
  "Why use a smaller model with LoRA on a Mac?",
  "What was the base model you tested with MLX?",
  "What did the anti-hallucination test verify?",
  "What did the summarization test verify?",
  "What is the main use case: “What have I written about X over 6 months?” enabling?",
  "What is meant by “longitudinal analysis” in MindLayer?",
  "What is meant by “pattern detection” in user writing?",
  "What is the high-level methodology list (ingestion → parsing → embeddings → storage → query → reasoning)?",
  "What is the core “second brain” value proposition?",
]

NEG = [
  "Who is the CEO of Apple?",
  "What is the capital of Brazil?",
  "Who won the FIFA World Cup 2022?",
  "What is the current price of Bitcoin today?",
  "Who is the Prime Minister of the UK right now?",
  "What is the weather in New York today?",
  "What is the square root of 987654?",
  "Give me the recipe for biryani.",
  "Who discovered penicillin?",
  "What is the full form of CPU?",
  "What year did the Titanic sink?",
  "Who is the founder of Tesla?",
  "What is the population of India in 2025?",
  "What is the deepest ocean trench?",
  "What are the symptoms of dengue?",
  "Solve: 47×83.",
  "Translate “good morning” into Japanese.",
  "What is the latest iPhone model?",
  "What is the GDP of France?",
  "Name 10 planets in the solar system.",
  "Who is the CEO of Google?",
  "What is the formula for variance?",
  "Define Keynesian economics.",
  "Write a poem about love.",
  "Tell me a joke.",
  "Explain quantum entanglement.",
  "What is the difference between HTTP and HTTPS?",
  "Give me Python code to reverse a list.",
  "What is the boiling point of mercury?",
  "Who is the author of “The Alchemist”?",
  "What is the current exchange rate USD to INR?",
  "When is the next solar eclipse?",
  "List the best tourist places in Paris.",
  "What are the causes of World War 1?",
  "What is the chemical formula of glucose?",
  "Who is the richest person in the world right now?",
  "What is the next SpaceX launch date?",
  "Who won the latest IPL season?",
  "What is the newest version of Python?",
  "What are today’s top tech news headlines?",
]

out = Path("backend/eval/eval_100.jsonl")
out.parent.mkdir(parents=True, exist_ok=True)

i = 1
with out.open("w", encoding="utf-8") as f:
    for q in POS:
        f.write(json.dumps({
            "id": i,
            "label": "POS",
            "expected": "ANSWER",
            "message": q,
            "top_k": 5,
            "kg_k": 10
        }, ensure_ascii=False) + "\n")
        i += 1
    for q in NEG:
        f.write(json.dumps({
            "id": i,
            "label": "NEG",
            "expected": "NOT_IN_MEMORY",
            "message": q,
            "top_k": 5,
            "kg_k": 10
        }, ensure_ascii=False) + "\n")
        i += 1

print("Wrote", out, "lines:", i-1)

