import datetime
import re
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed

import arxiv
import pandas as pd
from loguru import logger

from arxivrec.topic import Topic
from arxivrec.utils.fallback import fallback


class BaseFetcher(ABC):
    @abstractmethod
    def fetch(self, **kwargs) -> pd.DataFrame:
        pass


class ArxivFetcher(BaseFetcher):
    def __init__(
        self,
        topic: Topic,
        lookback_days: int = 1,
        max_results: int = 100,
    ):
        super().__init__()
        self.categories = topic.categories
        self.org_keywords = topic.org_keywords
        self.lookback_days = lookback_days
        self.max_results = max_results

    def _has_org_affiliation(self, paper_id: str, url: str) -> tuple[str, bool]:
        """Extract first-page text and check if any org keyword appears."""
        from arxivrec.dataset.parser import get_header_text

        try:
            header = get_header_text(url)
            pattern = "|".join(re.escape(k) for k in self.org_keywords)
            return paper_id, bool(re.search(pattern, header, re.IGNORECASE))
        except Exception as e:
            logger.warning(f"Could not parse {paper_id}: {e} — including by default")
            return paper_id, True

    def _filter_by_org(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Parsing {len(df)} PDFs for org affiliation (10 workers)...")

        keep_ids: set[str] = set()
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                executor.submit(self._has_org_affiliation, row["id"], row["url"]): row[
                    "id"
                ]
                for _, row in df.iterrows()
            }
            for future in as_completed(futures):
                paper_id, keep = future.result()
                if keep:
                    keep_ids.add(paper_id)

        filtered = df[df["id"].isin(keep_ids)]
        logger.info(
            f"Affiliation filter: {len(filtered)}/{len(df)} papers from known orgs"
        )
        return filtered

    @fallback("lookback_days", [2, 4])
    def fetch(self, **kwargs):
        """Fetches papers from specific categories within a time window."""

        # for fallback purpose only
        _lookback_days = kwargs.get("lookback_days", self.lookback_days)

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
            days=_lookback_days
        )

        results = []
        for result in client.results(search):
            # ArXiv results are sorted by date, so we can break early
            if result.published < threshold:
                break

            tmp = {
                "id": result.entry_id.split("/")[-1].split("v")[0],
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

        df = pd.DataFrame(results)

        if self.org_keywords and not df.empty:
            df = self._filter_by_org(df)

        return df
