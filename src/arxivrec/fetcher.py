import datetime
import logging
from abc import ABC, abstractmethod

import arxiv
import pandas as pd

logger = logging.getLogger(__name__)


class BaseFetcher(ABC):
    @abstractmethod
    def fetch(self, **kwargs) -> pd.DataFrame:
        pass


class ArxivFetcher(BaseFetcher):
    def __init__(
        self,
        categories: list[str] = ["cs.AI"],
        lookback_days: int = 1,
        max_results: int = 100,
    ):
        super().__init__()
        self.categories = categories
        self.lookback_days = lookback_days
        self.max_results = max_results

    def fetch(self):
        """
        Fetches papers from specific categories within a time window.
        """
        # 1. Build Query (e.g., "cat:cs.LG OR cat:cs.AI")
        query = " OR ".join([f"cat:{c}" for c in self.categories])

        client = arxiv.Client(
            page_size=100,
            delay_seconds=3.0,  # Be nice to ArXiv servers
            num_retries=3,
        )

        search = arxiv.Search(
            query=query,
            max_results=self.max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
        )

        # 2. Time Window
        threshold = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
            days=self.lookback_days
        )

        results = []
        for result in client.results(search):
            # ArXiv results are sorted by date, so we can break early
            if result.published < threshold:
                break

            tmp = {
                "id": result.entry_id.split("/")[-1],
                "title": result.title,
                "authors": [a.name for a in result.authors],
                "abstract": result.summary.replace("\n", " "),
                "published": result.published,
                "primary_category": result.primary_category,
                "url": result.pdf_url,
            }
            tmp["combined_text"] = f"Title: {tmp['title']}; Abstract: {tmp['abstract']}"
            results.append(tmp)

        since_time = threshold.strftime("%Y-%m-%d %H:%M")
        logger.info(f"Fetched {len(results)} new papers since {since_time}")
        return pd.DataFrame(results)
