import logging

from arxivrec.encoder import TextEncoder
from arxivrec.llm import OLLMRanker
from arxivrec.notification import send_email
from arxivrec.pipeline import OLLMPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    encoder = TextEncoder()
    ranker = OLLMRanker(model_name="llama3.2:3b")
    pipeline = OLLMPipeline(
        user_interest="I am interested in Retrieval-augmented generation (RAG).",
        max_paper_to_fetch=100,
        simsearch_top_k=10,
        encoder=encoder,
        ollm_ranker=ranker,
    )

    df_recommendation = pipeline.recommend()

    output_json = df_recommendation.to_json(orient="records")
    logger.info(f"Recommended articles: {output_json}")

    try:
        email_columns = ["title", "reasoning", "url"]
        html_table = df_recommendation[email_columns].to_html(
            index=False, render_links=True
        )

        full_body = f"""
        <html>
        <body>
            <h2>Today's ArXiv Recommendations 🐈</h2>
            <p>Based on your interest in RAG, here are the top 5 papers:</p>
            {html_table}
        </body>
        </html>
        """

        send_email(subject="📚 Your Daily ArXiv Digest", body_html=full_body)
        logger.info("Email notification sent successfully!")
    except Exception as e:
        logger.error("Failed to send email notification", exc_info=e)


if __name__ == "__main__":
    main()
