import logging
import sys

from arxivrec.encoder import TextEncoder
from arxivrec.fetcher import ArxivFetcher
from arxivrec.llm import OLLMRanker
from arxivrec.logger import setup_logging
from arxivrec.notification import EmailNotifier
from arxivrec.pipeline import OLLMPipeline
from arxivrec.topic import Topic

setup_logging()
logger = logging.getLogger(__name__)


def main():
    curr_topic = Topic(
        id="AI",
        description=(
            "Retrieval augmented generation (RAG),"
            "document parsing, table extraction",
        ),
        categories=["cs.AI"],
    )
    fetcher = ArxivFetcher(topic=curr_topic, lookback_days=4, max_results=100)
    encoder = TextEncoder(model_name="sentence-transformers/all-MiniLM-L6-v2")
    ranker = OLLMRanker(model_name="llama3.2:3b")
    notifier = EmailNotifier()

    pipeline = OLLMPipeline(
        topic=curr_topic,
        simsearch_top_k=10,
        fetcher=fetcher,
        encoder=encoder,
        ollm_ranker=ranker,
        notifier=notifier,
    )

    try:
        pipeline.recommend()
    except Exception as e:
        logger.exception(f"Error when running pipeline: {e}")
        sys.exit(1)

    try:
        pipeline.notify()
    except Exception as e:
        logger.exception(f"Error when sending notificatio: {e}")
        sys.exit(1)


if __name__ == "__main__":
    logger.info("Starting arXiv recommendation engine!")
    main()
