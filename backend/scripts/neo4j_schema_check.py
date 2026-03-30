"""
Neo4j schema diagnostics — run from repo root:
  python backend/scripts/neo4j_schema_check.py

Prints: claim count, labels, and relationship directions so you can
align kg_retrieval and claim extraction with the real graph.
"""
import os
from neo4j import GraphDatabase

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "mindlayer123")

def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as session:
        print("=== A. Claim node count ===")
        r = session.run("MATCH (c:Claim) RETURN count(c) AS claim_nodes").single()
        print("claim_nodes:", r["claim_nodes"] if r else 0)

        print("\n=== B. All labels (CALL db.labels) ===")
        labels = session.run("CALL db.labels() YIELD label RETURN label").data()
        for row in labels:
            print(" ", row["label"])

        print("\n=== C. Relationship directions (top 10) ===")
        rows = session.run("""
            MATCH (a)-[r]->(b)
            RETURN labels(a) AS from_labels, type(r) AS rel, labels(b) AS to_labels, count(*) AS n
            ORDER BY n DESC
            LIMIT 10
        """).data()
        for row in rows:
            print(" ", row["from_labels"], "-[%s]->" % row["rel"], row["to_labels"], "count =", row["n"])

        print("\n=== D. Entity -> Claim (your query direction) ===")
        rows = session.run("""
            MATCH (:Entity)-[r]->(:Claim)
            RETURN type(r) AS rel, count(*) AS n
            ORDER BY n DESC
        """).data()
        if not rows:
            print("  (no relationships)")
        for row in rows:
            print(" ", row["rel"], row["n"])

        print("\n=== E. Claim -> Entity (reversed direction) ===")
        rows = session.run("""
            MATCH (:Claim)-[r]->(:Entity)
            RETURN type(r) AS rel, count(*) AS n
            ORDER BY n DESC
        """).data()
        if not rows:
            print("  (no relationships)")
        for row in rows:
            print(" ", row["rel"], row["n"])

        print("\n=== F. Entity degree (Neo4j 5+ safe: COUNT { (e)--() }) ===")
        rows = session.run("""
            MATCH (e:Entity)
            RETURN e.name AS entity, COUNT { (e)--() } AS degree
            ORDER BY degree DESC
            LIMIT 15
        """).data()
        for row in rows:
            print(" ", row["entity"], "degree =", row["degree"])

    driver.close()
    print("\nDone. Use this to fix kg_retrieval / claim extraction if labels or direction differ.")

if __name__ == "__main__":
    main()
