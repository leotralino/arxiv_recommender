import argparse
import logging
import sys

from arxivrec.dataset.fetcher import ArxivFetcher
from arxivrec.engine.encoder import TextEncoder
from arxivrec.engine.llm import LLM_REGISTRY
from arxivrec.engine.ranker import LLMRanker
from arxivrec.notify.notification import NOTIFIER_REGISTRY
from arxivrec.pipeline import LLMPipeline
from arxivrec.topic import Topic
from arxivrec.utils.config_parse import load_config

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    logger.info(f"""
        ==========================================
        INITIALIZING ARXIV RECOMMENDER
        ------------------------------------------
        LLMs:      {', '.join(LLM_REGISTRY.show_available())}
        Notifiers: {', '.join(NOTIFIER_REGISTRY.show_available())}
        ==========================================
        """)

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

    client_name, client_args = next(iter(cfg["models"]["ranker"].items()))
    ranker = LLMRanker(client=LLM_REGISTRY[client_name](**client_args))

    notifier_list = []
    for type_param_pair in cfg["notifiers"]:
        for note_type, note_params in type_param_pair.items():
            note_class = NOTIFIER_REGISTRY[note_type]
            logger.info(f"Adding notifier: {note_class.__name__}")
            notifier_list.append(note_class(**{k: v for k, v in note_params.items()}))

    for curr_topic in topic_list:
        fetcher = ArxivFetcher(
            topic=curr_topic,
            lookback_days=cfg["pipeline"]["lookback_days"],
            max_results=cfg["pipeline"]["max_results"],
        )

        pipeline = LLMPipeline(
            topic=curr_topic,
            simsearch_top_k=cfg["pipeline"]["simsearch_top_k"],
            fetcher=fetcher,
            encoder=encoder,
            llm_ranker=ranker,
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
