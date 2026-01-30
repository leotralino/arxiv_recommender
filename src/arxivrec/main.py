import logging

from arxivrec.encoder import TextEncoder
from arxivrec.fetcher import fetch_recent_papers
from arxivrec.llm import fine_rank_with_llm
from arxivrec.notification import send_email

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

    output_json = refined_df.to_json(orient="records")
    print(output_json)

    try:
        # send notification email
        email_columns = ["title", "reasoning", "url"]  # Assuming you have a link column
        html_table = refined_df[email_columns].to_html(index=False, render_links=True)

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
    except Exception as e:
        logger.error("Failed to send email notification", exc_info=e)


if __name__ == "__main__":
    main()
