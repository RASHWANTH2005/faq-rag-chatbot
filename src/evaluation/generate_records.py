import json
from ragas import EvaluationDataset
from graph import app
import time

def generate_records():
    with open("data/eval_data.json") as f:
        test_cases = json.load(f)

    records = []
    for i, item in enumerate(test_cases ,start=1):
        start_time = time.time()

        result = app.invoke({"query": item["question"]})

        records.append({
            "user_input": item["question"],
            "response": result["answer"],
            "retrieved_contexts": [
                f"Question: {d.metadata['question']}\nAnswer: {d.metadata['answer']}"
                for d in result["docs"]
            ],
            "reference": item["ground_truth"],
        })
        print(f"record {i} processed in {time.time() - start_time:.2f}s")
        if i == 10:
            # time.sleep(60)  # Wait for 60 seconds after processing 25 records to avoid rate limiting
            break

    evaluation_dataset = EvaluationDataset.from_list(records)
    data = evaluation_dataset.to_list()

    with open("data/evaluation_dataset.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        print("Evaluation dataset saved to data/evaluation_dataset.json")


if __name__ == "__main__": 
    generate_records()