#!/usr/bin/env python3
"""
Simple script to test DBSCAN clustering on similar phrases
"""

import os
import sys
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Add the parent directory to the path so we can import from the memory framework
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    print("\n🧠 Testing DBSCAN Clustering on Similar Phrases\n")
    
    # Create test phrases - some similar, some different
    test_phrases = [
        "blue chair",
        "chair blue",
        "the chair is blue",
        "a blue colored chair",
        "red table",
        "table red",
        "green sofa",
        "sofa green"
    ]
    
    print("Test phrases:")
    for i, phrase in enumerate(test_phrases):
        print(f"  {i+1}. \"{phrase}\"")
    
    # Load the same encoder used in the memory framework
    print("\nLoading sentence transformer model...")
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Create embeddings
    print("Creating embeddings...")
    embeddings = encoder.encode(test_phrases)
    
    # Calculate similarity matrix to show relationships
    similarity_matrix = cosine_similarity(embeddings)
    
    print("\nSimilarity Matrix (cosine similarity):")
    print("    ", end="")
    for i in range(len(test_phrases)):
        print(f"{i+1:4d}", end="")
    print()
    
    for i in range(len(test_phrases)):
        print(f"{i+1:2d}: ", end="")
        for j in range(len(test_phrases)):
            print(f"{similarity_matrix[i][j]:.2f}", end=" ")
        print(f" | \"{test_phrases[i]}\"")
    
    # Test different eps values
    eps_values = [0.3, 0.4, 0.55, 0.7, 0.9]
    
    print("\nTesting different eps values for DBSCAN:")
    for eps in eps_values:
        dbscan = DBSCAN(eps=eps, min_samples=1)
        labels = dbscan.fit_predict(embeddings)
        
        # Group by cluster
        clusters = {}
        for i, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(i)
        
        print(f"\nDBSCAN with eps={eps}, min_samples=1:")
        for cluster_id, indices in clusters.items():
            print(f"  Cluster {cluster_id}: {indices} - ", end="")
            cluster_phrases = [test_phrases[idx] for idx in indices]
            print(", ".join([f"\"{phrase}\"" for phrase in cluster_phrases]))
        
        print(f"  Number of clusters: {len(clusters)}")

if __name__ == "__main__":
    main() 