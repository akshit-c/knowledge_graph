import os
import json
import uuid
import datetime
import sys
from pathlib import Path
from neo4j import GraphDatabase

# Ensure repo root is on sys.path so `import backend...` works
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.services.local_llm_mlx import generate

# -----------------------------
# Neo4j connection
# -----------------------------
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "mindlayer123")

MAX_CLAIMS_PER_CHUNK = 5

# -----------------------------
# Claim extraction prompt
# -----------------------------
CLAIM_PROMPT = """
You are extracting factual claims from personal notes.

Given the text below, extract up to 5 atomic claims.
Each claim must be grounded strictly in the text.

Return ONLY a valid JSON array.
Each item must contain:
- subject
- predicate
- object
- confidence (0.0 to 1.0)

If no clear claims exist, return [].

TEXT:
\"\"\"
{chunk_text}
\"\"\"
"""

def safe_parse_json(txt: str):
    """
    Parse JSON safely from LLM output.
    Returns list or empty list.
    """
    try:
        txt = txt.strip()
        # remove trailing junk
        if txt.startswith("```"):
            txt = txt.split("```")[1]
        data = json.loads(txt)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []

def main():
    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD)
    )

    with driver.session() as session:
        # Fetch chunks that do NOT yet have claims
        rows = session.run(
            """
            MATCH (c:Chunk)
            WHERE NOT (c)-[:SUPPORTS]->(:Claim)
            RETURN c.chunk_id AS chunk_id, c.text AS text
            """
        ).data()

        print(f"Chunks to process: {len(rows)}")

        total_claims = 0

        for row in rows:
            chunk_id = row["chunk_id"]
            text = (row["text"] or "").strip()

            if len(text) < 40:
                continue

            prompt = CLAIM_PROMPT.format(chunk_text=text[:2000])

            try:
                output = generate(prompt)
            except Exception as e:
                print(f"[WARN] LLM failed for chunk {chunk_id}: {e}")
                continue

            claims = safe_parse_json(output)[:MAX_CLAIMS_PER_CHUNK]

            for cl in claims:
                try:
                    subject = cl.get("subject", "").strip()
                    predicate = cl.get("predicate", "").strip()
                    obj = cl.get("object", "").strip()
                    confidence = float(cl.get("confidence", 0.0))

                    if not subject or not predicate or not obj:
                        continue

                    claim_id = str(uuid.uuid4())
                    created_at = datetime.datetime.utcnow().isoformat()

                    session.run(
                        """
                        MATCH (c:Chunk {chunk_id: $chunk_id})
                        MERGE (cl:Claim {claim_id: $claim_id})
                        SET cl.subject = $subject,
                            cl.predicate = $predicate,
                            cl.object = $object,
                            cl.confidence = $confidence,
                            cl.created_at = $created_at

                        MERGE (c)-[:SUPPORTS]->(cl)

                        MERGE (e:Entity {name: toLower($subject), type: "CONCEPT"})
                        MERGE (cl)-[:ABOUT]->(e)
                        """,
                        chunk_id=chunk_id,
                        claim_id=claim_id,
                        subject=subject,
                        predicate=predicate,
                        object=obj,
                        confidence=confidence,
                        created_at=created_at
                    )

                    total_claims += 1

                except Exception as e:
                    print(f"[WARN] Failed to store claim for chunk {chunk_id}: {e}")

        print(f"Done. Claims created: {total_claims}")

    driver.close()

if __name__ == "__main__":
    main()
