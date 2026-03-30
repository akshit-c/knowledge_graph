import os
import numpy as np

# Disable TensorFlow warnings (in case it's imported elsewhere)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_NO_TF'] = '1'

# Import sentence_transformers (uses PyTorch, not TensorFlow)
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

# Store the vectors and metadata in memory for now
DIMENSION = 384
vectors = []
metadata_store = []

def embed_and_store(content, metadata):
    """
    Embed content and store in memory.
    Returns the number of chunks stored.
    """
    # Validate content
    if not content or not content.strip():
        print(f"Warning: Empty content received for file {metadata.get('filename', 'unknown')}")
        return 0
    
    # Split content into chunks (512 characters each, with some overlap)
    chunk_size = 512
    chunks = []
    for i in range(0, len(content), chunk_size):
        chunk = content[i:i+chunk_size].strip()
        if chunk:  # Only add non-empty chunks
            chunks.append(chunk)
    
    if not chunks:
        print(f"Warning: No valid chunks created from content for file {metadata.get('filename', 'unknown')}")
        print(f"Content length: {len(content)}")
        return 0
    
    try:
        # Encode chunks to vectors
        vectors_chunks = model.encode(chunks)
        
        # Store vectors and metadata
        for i, (chunk, vector) in enumerate(zip(chunks, vectors_chunks)):
            vectors.append(vector)
            metadata_store.append({
                "text": chunk,
                "metadata": metadata,
                "vector_index": len(vectors) - 1
            })
        
        print(f"Successfully embedded {len(chunks)} chunks for file {metadata.get('filename', 'unknown')}")
        print(f"Total chunks in memory: {len(vectors)}")
        
        return len(chunks)
    except Exception as e:
        print(f"Error embedding content: {e}")
        return 0

def search_similar(query, top_k=5):
    if not vectors:
        return []
    
    query_vector = model.encode([query])
    
    # Calculate cosine similarity
    similarities = []
    for i, vector in enumerate(vectors):
        similarity = np.dot(query_vector[0], vector) / (np.linalg.norm(query_vector[0]) * np.linalg.norm(vector))
        similarities.append((similarity, i))
    
    # Sort by similarity and return top_k
    similarities.sort(reverse=True)
    results = []
    for similarity, idx in similarities[:top_k]:
        results.append({
            "text": metadata_store[idx]["text"],
            "metadata": metadata_store[idx]["metadata"],
            "similarity": float(similarity)
        })
    
    return results

def search_memory(query, top_k=5):
    """Alias for search_similar to maintain compatibility with query.py"""
    return search_similar(query, top_k)

def get_memory_stats():
    """Get statistics about the current memory"""
    filenames = [item["metadata"].get("filename", "unknown") for item in metadata_store]
    unique_files = set(filenames)
    
    return {
        "total_chunks": len(vectors),
        "total_documents": len(unique_files),
        "has_memory": len(vectors) > 0,
        "filenames": list(unique_files),
        "chunks_per_file": {filename: filenames.count(filename) for filename in unique_files}
    }

