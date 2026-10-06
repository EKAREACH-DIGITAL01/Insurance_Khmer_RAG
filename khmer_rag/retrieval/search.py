"""Dense and lexical search with reciprocal rank fusion."""

import time

import numpy as np

from khmer_rag.config import (
    BM25_TOP_K,
    RRF_K,
    TOP_K,
)
from khmer_rag.language import khmer_aware_tokenize


def encode_query(query, embedding_model, embedding_model_info):
    """Encodes the live user question as a QUERY. For e5-style models
    (multilingual-e5-large), prepends 'query: ' prefix to the raw question."""
    embed_start = time.time()
    if embedding_model_info.get("query_prefix"):
        query = embedding_model_info["query_prefix"] + query
    query_embedding = embedding_model.encode([query], normalize_embeddings=True)
    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32,
    )
    embed_time = time.time() - embed_start
    return query_embedding, embed_time


def search_faiss(query_embedding, index, chunks):
    search_start = time.time()
    scores, indices = index.search(
        query_embedding,
        k=TOP_K,
    )
    search_time = time.time() - search_start
    # FAISS pads small result sets with -1; never treat it as Python's last item.
    results = [
        (score, chunks[idx]) for score, idx in zip(scores[0], indices[0]) if 0 <= idx < len(chunks)
    ]
    return results, search_time


def search_bm25(query, bm25_index, chunks):
    search_start = time.time()
    tokenized_query = khmer_aware_tokenize(query)
    scores = bm25_index.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:BM25_TOP_K]
    results = [(float(scores[i]), chunks[i]) for i in top_indices]
    search_time = time.time() - search_start
    return results, search_time


def reciprocal_rank_fusion(result_lists, k=RRF_K):
    """Fuse ranked (score, chunk) lists by reciprocal rank.

    Fused scores are not cosine similarities. Relevance gates must continue
    to use raw FAISS scores."""
    fused_scores = {}
    chunk_lookup = {}
    for result_list in result_lists:
        for rank, (score, chunk) in enumerate(result_list, start=1):
            chunk_id = chunk["chunk_id"]
            chunk_lookup[chunk_id] = chunk
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (k + rank)

    ranked_chunk_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)
    return [(fused_scores[cid], chunk_lookup[cid]) for cid in ranked_chunk_ids]
