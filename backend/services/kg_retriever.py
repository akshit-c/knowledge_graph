from neo4j import GraphDatabase
from typing import List, Dict, Any
import os

# Neo4j connection (env overrides; defaults match local docker-compose)
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD", "mindlayer123")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

# Full-text KG retrieval: Claim nodes are indexed in "claim_text_ft".
# Evidence comes from Chunk nodes that SUPPORT a Claim.
KG_QUERY = """
CALL db.index.fulltext.queryNodes("claim_text_ft", $q)
YIELD node AS c, score AS ft
WHERE c:Claim

OPTIONAL MATCH (ch:Chunk)-[:SUPPORTS]->(c)
WITH c, ft,
     collect({
       chunk_id: ch.chunk_id,
       doc_id: ch.doc_id,
       text: ch.text
     })[0..$evidence_k] AS evidence

WITH c, ft, evidence,
     size(evidence) AS support,
     coalesce(c.confidence, 0.6) AS conf,
     CASE
       WHEN c.created_at IS NULL OR c.created_at = "" THEN datetime()
       ELSE datetime(c.created_at)
     END AS created

WITH c, ft, evidence, support, conf, created,
     (1.0 / (1.0 + duration.between(created, datetime()).days)) AS recency
WITH c, ft, evidence, support, conf, recency,
     (2.5*ft + 1.8*support + 1.2*conf + 1.5*recency) AS kg_score

RETURN
  c.text AS claim,
  round(kg_score, 3) AS kg_score,
  support,
  evidence
ORDER BY kg_score DESC
LIMIT $k
"""


def fetch_kg_context(question: str, k: int = 5, evidence_k: int = 2) -> List[Dict[str, Any]]:
    """
    Run KG_QUERY to get top-k claims plus their supporting evidence chunks.
    """
    q = (question or "").strip()
    if not q:
        return []

    with driver.session() as session:
        res = session.run(
            KG_QUERY,
            q=q,
            k=k,
            evidence_k=evidence_k,
        )
        return [r.data() for r in res]


def build_kg_context(kg_results: List[Dict[str, Any]]) -> str:
    """
    Turn KG results into a labeled, auditable context block for the LLM.
    """
    blocks: List[str] = []

    for i, r in enumerate(kg_results, 1):
        claim_text = r.get("claim") or ""
        blocks.append(f"CLAIM {i}: {claim_text}")

        evidence_list = r.get("evidence") or []
        for j, ev in enumerate(evidence_list, 1):
            ev_text = (ev.get("text") or "").strip()
            blocks.append(
                f"  Evidence {i}.{j}:\n"
                f"  {ev_text}"
            )

    return "\n\n".join(blocks)

