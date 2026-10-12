def rrf_fusion(
    dense_docs,
    bm25_docs,
    rrf_k=60
):

    scores = {}

    documents = {}

    # =====================================
    # Dense results
    # =====================================

    for rank, doc in enumerate(
        dense_docs,
        start=1
    ):

        doc_id = doc.metadata["id"]

        score = 1 / (
            rrf_k + rank
        )

        scores[doc_id] = (
            scores.get(doc_id, 0)
            + score
        )

        documents[doc_id] = doc

    # =====================================
    # BM25 results
    # =====================================

    for rank, doc in enumerate(
        bm25_docs,
        start=1
    ):

        doc_id = doc.metadata["id"]

        score = 1 / (
            rrf_k + rank
        )

        scores[doc_id] = (
            scores.get(doc_id, 0)
            + score
        )

        documents[doc_id] = doc

    # =====================================
    # Sort by RRF score
    # =====================================

    ranked_ids = sorted(
        scores,
        key=scores.get,
        reverse=True
    )

    # =====================================
    # Return documents in ranked order
    # =====================================

    return [
        documents[doc_id]
        for doc_id in ranked_ids
    ]