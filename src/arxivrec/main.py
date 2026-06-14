import argparse
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

from arxivrec.dataset.fetcher import ArxivFetcher, BaseFetcher
from arxivrec.dataset.hf_fetcher import HFDailyPapersFetcher
from arxivrec.engine.encoder import TextEncoder
from arxivrec.engine.llm import LLM_REGISTRY
from arxivrec.engine.ranker import LLMRanker
from arxivrec.notify.notification import NOTIFIER_REGISTRY
from arxivrec.pipeline import LLMPipeline, build_digest_html
from arxivrec.topic import Topic
from arxivrec.utils.config_parse import load_config
from arxivrec.utils.logger import show_registry_table, show_topic_table


def main():
    parser = argparse.ArgumentParser()
    _default_config = Path(__file__).parent.parent / "config.yaml"
    parser.add_argument("--config", type=str, default=str(_default_config))
    args = parser.parse_args()

    cfg = load_config(args.config)

    show_registry_table(LLM_REGISTRY, NOTIFIER_REGISTRY)

    topic_list = []
    for topic_data in cfg["topic"]:
        logger.info(f"Processing Topic: {topic_data['id']}")
        topic_list.append(
            Topic(
                id=topic_data["id"],
                description=topic_data["description"],
                categories=topic_data.get("categories", []),
                org_keywords=topic_data.get("org_keywords", []),
                source=topic_data.get("source", "arxiv"),
            )
        )

    show_topic_table(topic_list=topic_list)

    encoder = TextEncoder(model_name=cfg["models"]["encoder"])

    client_name, client_args = next(iter(cfg["models"]["ranker"].items()))
    ranker = LLMRanker(client=LLM_REGISTRY[client_name](**client_args))

    notifier_list = []
    for type_param_pair in cfg["notifiers"]:
        for note_type, note_params in type_param_pair.items():
            note_class = NOTIFIER_REGISTRY[note_type]
            logger.info(f"Adding notifier: {note_class.__name__}")
            notifier_list.append(note_class(**{k: v for k, v in note_params.items()}))

    all_results: dict[str, pd.DataFrame] = {}

    for curr_topic in topic_list:
        topic_data = next(t for t in cfg["topic"] if t["id"] == curr_topic.id)

        fetcher: BaseFetcher
        if curr_topic.source == "huggingface":
            fetcher = HFDailyPapersFetcher(
                topic=curr_topic,
                min_upvotes=topic_data.get("min_upvotes", 0),
            )
        else:
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
            notifier_list=[],
        )

        logger.info(f"Running pipeline for topic: {curr_topic.id}")

        try:
            df = pipeline.recommend()
            if df is not None and not df.empty:
                all_results[curr_topic.id] = df
        except Exception as e:
            logger.exception(f"Error running pipeline for topic '{curr_topic.id}': {e}")

    if not all_results:
        logger.error("No recommendations generated for any topic.")
        sys.exit(1)

    try:
        digest_html = build_digest_html(topic_list, all_results)
    except Exception as e:
        logger.exception(f"Failed to build digest: {e}")
        sys.exit(1)

    try:
        report_path = Path("report.html")
        report_path.write_text(digest_html, encoding="utf-8")
        logger.info(f"Report saved to {report_path.resolve()}")
    except Exception as e:
        logger.exception(f"Failed to save report.html: {e}")

    for notifier in notifier_list:
        try:
            notifier.notify(subject="📚 Your Daily ArXiv Digest", body_html=digest_html)
            logger.info(f"{notifier} digest sent successfully!")
        except Exception as e:
            logger.warning(
                f"{notifier.__class__.__name__} not configured locally"
                f" — skipping. ({e})"
            )


if __name__ == "__main__":
    logger.info("Starting arXiv recommendation engine!")
    main()
