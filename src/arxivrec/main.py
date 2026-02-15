import argparse
import logging
import sys

import yaml
from dotenv import load_dotenv

from arxivrec.dataset.fetcher import ArxivFetcher
from arxivrec.engine.encoder import TextEncoder
from arxivrec.engine.ranker import OLLMRanker
from arxivrec.notify.notification import EmailNotifier, RSSNotifier, SlackNotifier
from arxivrec.pipeline import OLLMPipeline
from arxivrec.topic import Topic
from arxivrec.utils.logger import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

load_dotenv()


def load_config(config_path: str = "config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    topic_list = []
    for topic_data in cfg["topic"]:
        logger.info(f"Processing Topic: {topic_data['id']}")

        topic_list.append(
            Topic(
                id=topic_data["id"],
                description=topic_data["description"],
                categories=topic_data["categories"],
            )
        )

    encoder = TextEncoder(model_name=cfg["models"]["encoder"])
    ranker = OLLMRanker(model_name=cfg["models"]["ranker"])

    notifier_list = []
    NOTIFIER_NAME_MAPPING = {
        "email": EmailNotifier,
        "slack": SlackNotifier,
        "rss": RSSNotifier,
    }

    for type_param_pair in cfg["notifiers"]:
        for note_type, note_params in type_param_pair.items():
            note_class = NOTIFIER_NAME_MAPPING[note_type]
            logger.info(f"Adding notifier: {note_class.__name__}")
            notifier_list.append(note_class(**{k: v for k, v in note_params.items()}))

    for curr_topic in topic_list:
        fetcher = ArxivFetcher(
            topic=curr_topic,
            lookback_days=cfg["pipeline"]["lookback_days"],
            max_results=cfg["pipeline"]["max_results"],
        )

        pipeline = OLLMPipeline(
            topic=curr_topic,
            simsearch_top_k=cfg["pipeline"]["simsearch_top_k"],
            fetcher=fetcher,
            encoder=encoder,
            ollm_ranker=ranker,
            notifier_list=notifier_list,
        )

        logger.info(f"Created pipeline: {pipeline}")

        try:
            pipeline.recommend()
        except Exception as e:
            logger.exception(f"Error when running pipeline: {e}")
            sys.exit(1)

        try:
            pipeline.notify()
        except Exception as e:
            logger.exception(f"Error when sending notification: {e}")
            sys.exit(1)


if __name__ == "__main__":
    logger.info("Starting arXiv recommendation engine!")
    main()
