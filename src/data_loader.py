import json

from langchain_core.documents import Document




def load_documents():

    with open(
        "data/faqs.json",
        "r",
        encoding="utf-8"
    ) as f:
        faqs = json.load(f)

    docs = []

    for i, faq in enumerate(faqs):

        doc = Document(
            page_content=f"Question: {faq['question']}",
            metadata={
                "id": str(i),
                "question": faq["question"],
                "answer": faq["answer"]
            }
        )

        docs.append(doc)

    return docs