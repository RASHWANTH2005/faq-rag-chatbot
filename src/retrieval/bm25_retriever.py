from rank_bm25 import BM25Okapi


class BM25Retriever:

    def __init__(self, docs):

        self.docs = docs

        # --------------------------------
        # Tokenize every document
        # --------------------------------

        tokenized_docs = [
            doc.page_content.lower().split()
            for doc in docs
        ]

        # --------------------------------
        # Build BM25 index
        # --------------------------------

        self.bm25 = BM25Okapi(
            tokenized_docs
        )

    def retrieve(
        self,
        query,
        k=20
    ):

        # --------------------------------
        # Tokenize query
        # --------------------------------

        tokenized_query = query.lower().split()

        # --------------------------------
        # Get BM25 score for every document
        # --------------------------------

        scores = self.bm25.get_scores(
            tokenized_query
        )

        # --------------------------------
        # Get document indices sorted
        # by BM25 score
        # --------------------------------

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )

        # --------------------------------
        # Select top K
        # --------------------------------

        top_indices = ranked_indices[:k]

        # --------------------------------
        # Return actual documents
        # --------------------------------

        return [
            self.docs[i]
            for i in top_indices
        ]