from neo4j import GraphDatabase
from typing import List, Dict, Any
import os
import re

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "mindlayer123")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def _entity_names_from_question(question: str, max_tokens: int = 15) -> List[str]:
    """Normalize question to lowercase entity-name candidates (no duplicates)."""
    q = (question or "").strip()
    if not q:
        return []
    tokens = [
        t.lower()
        for t in re.sub(r"[^\w\s]", " ", q).split()
        if len(t) >= 2
    ]
    return list(dict.fromkeys(tokens))[:max_tokens]

def fetch_claims_for_question(question: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Entity-first KG retrieval. Supports both relationship directions so it never
    returns 0 if the graph is (Claim)->(Entity) or (Entity)->(Claim). Entity names
    from the question are normalized to lowercase. chunk_ids from (Chunk)-[:SUPPORTS]->(Claim).
    """
    q = (question or "").strip()
    if not q:
        return []

    entity_names = _entity_names_from_question(q)
    if not entity_names:
        return []

    # Resilient: accept either (c)-[:MENTIONS|ABOUT]->(e) or (e)-[:MENTIONS|ABOUT]->(c)
    cypher = """
    MATCH (e:Entity)
    WHERE toLower(e.name) IN $entity_names
       OR (e.canonical_name IS NOT NULL AND toLower(e.canonical_name) IN $entity_names)
    MATCH (c:Claim)
    WHERE (c)-[:MENTIONS|ABOUT]->(e) OR (e)-[:MENTIONS|ABOUT]->(c)
    OPTIONAL MATCH (ch:Chunk)-[:SUPPORTS]->(c)
    WITH c, count(DISTINCT e) AS support, collect(DISTINCT ch.chunk_id) AS chunk_ids
    RETURN
      c.subject AS subject,
      c.predicate AS predicate,
      c.object AS object,
      c.text AS text,
      coalesce(c.confidence, 0.5) AS confidence,
      support AS hits,
      chunk_ids AS chunk_ids,
      toFloat(support) AS ft_score
    ORDER BY support DESC, confidence DESC
    LIMIT $k
    """

    with driver.session() as session:
        rows = session.run(
            cypher,
            entity_names=entity_names,
            k=limit,
        ).data()

    out = []
    for r in rows:
        s = (r.get("subject") or "").strip()
        p = (r.get("predicate") or "").strip()
        o = (r.get("object") or "").strip()
        triple = f"({s}) -[{p}]-> ({o})"
        raw_chunk_ids = r.get("chunk_ids") or []
        chunk_ids = [x for x in raw_chunk_ids if x is not None]
        out.append({
            "triple": triple,
            "hits": int(r.get("hits") or 1),
            "confidence": float(r.get("confidence") or 0.5),
            "chunk_ids": chunk_ids,
            "ft_score": float(r.get("ft_score") or 0.0),
        })

    return out
