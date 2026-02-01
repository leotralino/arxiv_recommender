import logging
import sys

from arxivrec.encoder import TextEncoder
from arxivrec.fetcher import ArxivFetcher
from arxivrec.llm import OLLMRanker
from arxivrec.logger import setup_logging
from arxivrec.notification import EMAIL_TEMPLATE, send_email
from arxivrec.pipeline import OLLMPipeline

setup_logging()
logger = logging.getLogger(__name__)


def main():
    user_interest = "I am interested in Retrieval-augmented generation (RAG)."
    fetcher = ArxivFetcher(categories=["cs.AI"], lookback_days=4, max_results=100)
    encoder = TextEncoder(model_name="sentence-transformers/all-MiniLM-L6-v2")
    ranker = OLLMRanker(model_name="llama3.2:3b")
    pipeline = OLLMPipeline(
        user_interest=user_interest,
        simsearch_top_k=10,
        fetcher=fetcher,
        encoder=encoder,
        ollm_ranker=ranker,
    )

    try:
        df_recommendation = pipeline.recommend()
    except Exception as e:
        logger.exception(f"Error when running pipeline: {e}")
        sys.exit(1)

    output_json = df_recommendation.to_json(orient="records")
    logger.info(f"Recommended articles: {output_json}")

    try:
        email_columns = ["title", "reasoning", "url"]
        html_table = df_recommendation[email_columns].to_html(
            index=False, render_links=True
        )

        email_full_body = EMAIL_TEMPLATE.format(
            my_interest=user_interest, input_html_table=html_table
        )
        send_email(subject="📚 Your Daily ArXiv Digest", body_html=email_full_body)
        logger.info("Email notification sent successfully!")
    except Exception:
        logger.exception("Failed to send email notification!")
        sys.exit(1)


if __name__ == "__main__":
    logger.info("Starting arXiv recommendation engine!")
    main()
