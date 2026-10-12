from sentence_transformers import CrossEncoder


class CrossEncoderReranker:

    def __init__(
        self,
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):

        self.model = CrossEncoder(
            model_name
        )

    def rerank(
        self,
        query,
        documents,
        top_k=5
    ):

        # --------------------------------
        # Create query-document pairs
        # --------------------------------

        pairs = [
            (
                query,
                f"Question: {doc.metadata['question']}\n"
                f"Answer: {doc.metadata['answer']}"
            )
            for doc in documents
        ]
        # --------------------------------
        # Cross encoder scores
        # --------------------------------

        scores = self.model.predict(
            pairs
        )

        # --------------------------------
        # Combine docs + scores
        # --------------------------------

        ranked = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True
        )

        # --------------------------------
        # Return top K
        # --------------------------------

        return ranked[:top_k]