import os
import re
from collections import Counter
from neo4j import GraphDatabase

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "mindlayer123")

# Simple, paper-friendly baseline: extract "candidate entities" as
# - Capitalized phrases (e.g., "MindLayer", "Personal AI Memory")
# - Important nouns-ish tokens longer than 3 chars
CAP_PHRASE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b")
WORD = re.compile(r"\b[a-zA-Z][a-zA-Z\-]{3,}\b")

STOP = set("""
the a an and or of in on for to with is are was were be been being this that these those
as by from at into it its you your we our they their not only also can will would should
""".split())

def extract_candidates(text: str):
    cands = []

    # 1) Capitalized phrases
    for m in CAP_PHRASE.findall(text):
        name = m.strip()
        if len(name) >= 4 and name.lower() not in STOP:
            cands.append(("PROPER", name))

    # 2) Keywords (lowercase words)
    for w in WORD.findall(text):
        lw = w.lower()
        if lw in STOP: 
            continue
        # Keep some technical-ish words
        if len(lw) >= 5:
            cands.append(("TERM", lw))

    return cands

def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    with driver.session() as session:
        # Pull chunks from Neo4j
        rows = session.run(
            "MATCH (c:Chunk) RETURN c.chunk_id AS chunk_id, c.text AS text"
        ).data()

        print(f"Chunks fetched: {len(rows)}")

        written_edges = 0
        written_entities = 0

        for r in rows:
            chunk_id = r["chunk_id"]
            text = r["text"] or ""
            cands = extract_candidates(text)

            # Count and keep top N per chunk to avoid noise
            cnt = Counter([name for _, name in cands])
            top = cnt.most_common(12)

            for name, freq in top:
                # heuristic typing
                etype = "TERM"
                if any(ch.isupper() for ch in name) and " " in name:
                    etype = "CONCEPT"
                elif any(ch.isupper() for ch in name):
                    etype = "PROPER"

                confidence = min(1.0, 0.2 + 0.1 * freq)

                session.run(
                    """
                    MATCH (c:Chunk {chunk_id: $chunk_id})
                    MERGE (e:Entity {name: $name, type: $type})
                    MERGE (c)-[r:MENTIONS]->(e)
                    SET r.confidence = $confidence
                    """,
                    chunk_id=chunk_id,
                    name=name,
                    type=etype,
                    confidence=confidence
                )
                written_edges += 1

        # quick counts
        res = session.run("MATCH (e:Entity) RETURN count(e) AS n").single()
        written_entities = res["n"]

    driver.close()
    print("Done.")
    print("Entities in graph:", written_entities)
    print("MENTIONS edges written:", written_edges)

if __name__ == "__main__":
    main()
