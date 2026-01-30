import logging

from arxivrec.encoder import TextEncoder
from arxivrec.fetcher import fetch_recent_papers
from arxivrec.llm import fine_rank_with_llm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    df = fetch_recent_papers(max_results=100)

    encoder = TextEncoder()

    my_interest_embedding = encoder.encode(["I am interested in Retrieval-augmented generation (RAG)."])
    content_embedding = encoder.encode(df["combined_text"].tolist())
    top_k_indices = encoder.get_top_k_similar(my_interest_embedding, content_embedding, k=10)

    top_df = df.iloc[top_k_indices]

    refined_df = fine_rank_with_llm(top_df)
    print(refined_df[["id", "title", "Quality Score"]])
    return refined_df.to_json(orient="records")


if __name__ == "__main__":
    main()
