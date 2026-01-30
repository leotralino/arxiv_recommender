import datetime
import logging

import arxiv
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fetch_recent_papers(categories: list[str] = ["cs.AI"], days: int = 1, max_results=100):
    """
    Fetches papers from specific categories within a time window.
    """
    # 1. Build Query (e.g., "cat:cs.LG OR cat:cs.AI")
    query = " OR ".join([f"cat:{c}" for c in categories])

    client = arxiv.Client(
        page_size=100,
        delay_seconds=3.0,  # Be nice to ArXiv servers
        num_retries=3,
    )

    search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.SubmittedDate)

    # 2. Time Window
    threshold = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days)

    results = []
    for result in client.results(search):
        # ArXiv results are sorted by date, so we can break early
        if result.published < threshold:
            break

        results.append(
            {
                "id": result.entry_id.split("/")[-1],
                "title": result.title,
                "authors": [a.name for a in result.authors],
                "abstract": result.summary.replace("\n", " "),
                "published": result.published,
                "primary_category": result.primary_category,
                "url": result.pdf_url,
            }
        )

    since_time = threshold.strftime("%Y-%m-%d %H:%M")
    print(f"Fetched {len(results)} new papers since {since_time}")
    return pd.DataFrame(results)
