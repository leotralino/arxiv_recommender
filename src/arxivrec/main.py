import argparse
import logging
import sys

import yaml

from arxivrec.encoder import TextEncoder
from arxivrec.fetcher import ArxivFetcher
from arxivrec.llm import OLLMRanker
from arxivrec.logger import setup_logging
from arxivrec.notification import EmailNotifier
from arxivrec.pipeline import OLLMPipeline
from arxivrec.topic import Topic

setup_logging()
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    curr_topic = Topic(
        id=cfg["topic"]["id"],
        description=cfg["topic"]["description"],
        categories=cfg["topic"]["categories"],
    )

    fetcher = ArxivFetcher(
        topic=curr_topic,
        lookback_days=cfg["pipeline"]["lookback_days"],
        max_results=cfg["pipeline"]["max_results"],
    )

    encoder = TextEncoder(model_name=cfg["models"]["encoder"])
    ranker = OLLMRanker(model_name=cfg["models"]["ranker"])
    notifier = EmailNotifier()

    pipeline = OLLMPipeline(
        topic=curr_topic,
        simsearch_top_k=cfg["pipeline"]["simsearch_top_k"],
        fetcher=fetcher,
        encoder=encoder,
        ollm_ranker=ranker,
        notifier=notifier,
    )

    logger.info(f"Created pipeline: {pipeline}")

    try:
        pipeline.recommend()
    except Exception as e:
        logger.exception(f"Error when running pipeline: {e}")
        sys.exit(1)

    try:
        if cfg["notifiers"]["email"]:
            pipeline.notify()
    except Exception as e:
        logger.exception(f"Error when sending notificatio: {e}")
        sys.exit(1)


if __name__ == "__main__":
    logger.info("Starting arXiv recommendation engine!")
    main()
