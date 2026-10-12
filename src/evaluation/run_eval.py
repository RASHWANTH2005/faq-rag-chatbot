import json
import os
from ragas import EvaluationDataset, evaluate
from ragas.run_config import RunConfig
from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall, ResponseRelevancy
import time
import pandas as pd


from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

judge_llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

#NOTE :: embeddings are not needed for these metrics 

# judge_embeddings = OllamaEmbeddings(
#             model="nomic-embed-text",
#             base_url="http://localhost:11434"
#         )   



def run_evaluation() -> pd.DataFrame:

    with open("data/evaluation_dataset.json", "r", encoding="utf-8") as f:
        records = json.load(f)

    # limit = int(os.getenv("EVAL_LIMIT", "0"))   # optional smoke-test slice
    # if limit:
    #     records = records[:limit]

    evaluation_dataset = EvaluationDataset.from_list(records)
    BATCH_SIZE = 4
    batch_frames = []

    for start in range(0, len(evaluation_dataset), BATCH_SIZE):
        st_time = time.time()
        end = min(start + BATCH_SIZE, len(evaluation_dataset))
        batch = evaluation_dataset[start:end]

        print(f"\nEvaluating records {start + 1} to {end}")

        result = evaluate(
            batch,
            metrics=[
                Faithfulness(),
                LLMContextPrecisionWithReference(),
            ],
            llm=LangchainLLMWrapper(judge_llm),
            # embeddings=LangchainEmbeddingsWrapper(judge_embeddings), 
            run_config=RunConfig(max_workers=1, timeout=180),
        )

        batch_frames.append(result.to_pandas())   # one row per record

        if end < len(evaluation_dataset):
            elapsed_time = time.time() - st_time
            wait_time = max(0, 60 - elapsed_time)
            print(f"Batch took {elapsed_time:.2f} seconds")
            print(f"Waiting {wait_time:.2f} seconds...")
            time.sleep(wait_time + 2)

    df = pd.concat(batch_frames, ignore_index=True)
    print(df.columns.tolist())
    df.to_csv("eval_results.csv", index=False)
    return df