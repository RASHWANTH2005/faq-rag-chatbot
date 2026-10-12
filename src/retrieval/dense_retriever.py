from langchain_chroma import Chroma


class DenseRetriever:

    def __init__(self, chroma):
        self.chroma = chroma

    def retrieve(self, query, k=1):

        docs = self.chroma.similarity_search(
            query,
            k=k
        )

        return docs