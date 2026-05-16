"""Evaluation utilities for retrieval models."""
import torch
import torch.nn.functional as F
from tqdm import tqdm
from beir.retrieval.evaluation import EvaluateRetrieval
import numpy as np


  
def retrieval_evaluate(query_embeddings, doc_embeddings, qrels, top_k=100):
    # Compute similarities using vectorized operations
    results = _compute_similarities_batched(query_embeddings, doc_embeddings, top_k)
    # Compute metrics
    evaluator = EvaluateRetrieval()
    filtered_qrels = {qid: {qrels[qid][0]:1} for qid in results.keys() if qid in qrels}
    ndcg, _map, recall, precision = evaluator.evaluate(filtered_qrels, results, [1, 3, 5, 10, 100])
    return results, ndcg, _map, recall, precision


def _compute_similarities_batched(query_embeddings, doc_embeddings, top_k):
    """Compute similarities using vectorized operations - super fast!"""
    print("⚡ Computing similarities with vectorized operations...")
    # Convert to matrices for vectorized computation
    query_ids = list(query_embeddings.keys())
    doc_ids = list(doc_embeddings.keys())
    # Stack embeddings into matrices
    query_matrix = np.stack([query_embeddings[qid] for qid in query_ids])
    doc_matrix = np.stack([doc_embeddings[did] for did in doc_ids])
    # Compute all similarities at once (vectorized - very fast!)
    # Normalize embeddings for cosine similarity
    query_matrix = query_matrix / np.linalg.norm(query_matrix, axis=1, keepdims=True)
    doc_matrix = doc_matrix / np.linalg.norm(doc_matrix, axis=1, keepdims=True)
    similarity_matrix = np.dot(query_matrix, doc_matrix.T)
    # Process results
    results = {}
    for i, query_id in enumerate(tqdm(query_ids, desc="Processing results")):
        # Get similarities for this query
        query_similarities = similarity_matrix[i]
        # Adjust top_k if the number of documents is smaller than top_k
        current_top_k = min(top_k, len(query_similarities))
        # Get top-k documents
        top_indices = np.argpartition(query_similarities, -current_top_k)[-current_top_k:]
        top_indices = top_indices[np.argsort(query_similarities[top_indices])[::-1]]
        # Build results dict
        results[query_id] = {}
        for idx in top_indices:
            doc_id = doc_ids[idx]
            score = float(query_similarities[idx])
            results[query_id][doc_id] = score
    return results


def compute_cosine_similarities(document_embeddings: torch.Tensor, 
                                      query_embeddings: torch.Tensor) -> torch.Tensor:
    document_normalized = torch.nn.functional.normalize(document_embeddings, p=2, dim=1)
    query_normalized = torch.nn.functional.normalize(query_embeddings, p=2, dim=1)
    return torch.diag(torch.matmul(query_normalized, document_normalized.T))
