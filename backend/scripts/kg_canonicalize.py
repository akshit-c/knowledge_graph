import re
import os
from neo4j import GraphDatabase

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "mindlayer123")

def canon(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[\s\-_]+", " ", s)          # normalize spaces
    s = re.sub(r"[^a-z0-9 ]+", "", s)        # remove punctuation
    s = re.sub(r"\s+", " ", s).strip()
    return s

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def main():
    with driver.session() as session:
        # 1) Add canonical_name to all Entity nodes (use elementId for Neo4j 5.x)
        rows = session.run("MATCH (e:Entity) RETURN elementId(e) AS id, e.name AS name").data()
        print(f"Entities found: {len(rows)}")

        for r in rows:
            cid = r["id"]
            name = r["name"]
            ckey = canon(name)
            session.run(
                "MATCH (e:Entity) WHERE elementId(e)=$id SET e.canonical_name=$ckey",
                id=cid, ckey=ckey
            )

        # 2) Merge duplicates by canonical_name
        groups = session.run("""
            MATCH (e:Entity)
            WHERE e.canonical_name IS NOT NULL AND e.canonical_name <> ""
            WITH e.canonical_name AS k, collect(e) AS nodes
            WHERE size(nodes) > 1
            RETURN k, [n IN nodes | elementId(n)] AS ids, [n IN nodes | n.name] AS names
        """).data()

        print(f"Duplicate groups: {len(groups)}")

        merged = 0
        for g in groups:
            ids = g["ids"]
            # choose the first node as the "keeper"
            keep_id = ids[0]
            merge_ids = ids[1:]

            for mid in merge_ids:
                # move incoming relationships
                session.run("""
                    MATCH (a)-[r]->(b:Entity)
                    WHERE elementId(b)=$mid AND elementId(b)<>$keep
                    MATCH (k:Entity) WHERE elementId(k)=$keep
                    MERGE (a)-[r2:MENTIONS]->(k)
                    ON CREATE SET r2.count = coalesce(r.count,1)
                    ON MATCH SET r2.count = coalesce(r2.count,1) + coalesce(r.count,1)
                    DELETE r
                """, mid=mid, keep=keep_id)

                # move outgoing relationships
                session.run("""
                    MATCH (b:Entity)-[r]->(a)
                    WHERE elementId(b)=$mid AND elementId(b)<>$keep
                    MATCH (k:Entity) WHERE elementId(k)=$keep
                    MERGE (k)-[r2:MENTIONS]->(a)
                    ON CREATE SET r2.count = coalesce(r.count,1)
                    ON MATCH SET r2.count = coalesce(r2.count,1) + coalesce(r.count,1)
                    DELETE r
                """, mid=mid, keep=keep_id)

                # delete the duplicate node
                session.run("MATCH (e:Entity) WHERE elementId(e)=$mid DETACH DELETE e", mid=mid)
                merged += 1

        print(f"Merged entity nodes: {merged}")

        # 3) Build claim.text for better full-text search
        session.run("""
            MATCH (c:Claim)
            SET c.text = coalesce(c.subject,"") + " " + coalesce(c.predicate,"") + " " + coalesce(c.object,"")
        """)
        print("Updated Claim.text")

    driver.close()

if __name__ == "__main__":
    main()
