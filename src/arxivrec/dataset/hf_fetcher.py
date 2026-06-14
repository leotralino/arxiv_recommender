from datetime import date, timedelta

import pandas as pd
import requests
from loguru import logger

from arxivrec.dataset.fetcher import BaseFetcher
from arxivrec.topic import Topic

_HF_API = "https://huggingface.co/api/daily_papers"


class HFDailyPapersFetcher(BaseFetcher):
    def __init__(self, topic: Topic, min_upvotes: int = 0, lookback_days: int = 3):
        self.topic = topic
        self.min_upvotes = min_upvotes
        self.lookback_days = lookback_days

    def _fetch_for_date(self, d: date) -> list[dict]:
        response = requests.get(_HF_API, params={"date": d.isoformat()}, timeout=15)
        response.raise_for_status()
        return response.json()

    def fetch(self, **kwargs) -> pd.DataFrame:
        # Walk back until we find a day with papers (handles weekends/gaps)
        raw: list[dict] = []
        for days_ago in range(self.lookback_days):
            d = date.today() - timedelta(days=days_ago)
            raw = self._fetch_for_date(d)
            if raw:
                logger.info(f"HF Daily Papers: found {len(raw)} papers for {d}")
                break
        else:
            logger.warning(
                f"HF Daily Papers: no papers found in last {self.lookback_days} days"
            )
            return pd.DataFrame()

        results = []
        for item in raw:
            paper = item["paper"]
            upvotes = paper.get("upvotes", 0)

            if upvotes < self.min_upvotes:
                continue

            arxiv_id = paper["id"]
            authors = [
                a["name"] for a in paper.get("authors", []) if not a.get("hidden")
            ]
            org_info = paper.get("organization") or {}
            org = org_info.get("fullname", "")

            results.append(
                {
                    "id": arxiv_id,
                    "title": paper["title"],
                    "authors": authors,
                    "abstract": paper.get("summary", "").replace("\n", " "),
                    "published": paper.get("publishedAt", ""),
                    "primary_category": "",
                    "url": f"https://arxiv.org/pdf/{arxiv_id}",
                    "upvotes": upvotes,
                    "org": org,
                }
            )

        df = pd.DataFrame(results)
        if not df.empty:
            df["combined_text"] = df.apply(
                lambda r: f"Title: {r['title']}; Abstract: {r['abstract']}", axis=1
            )

        suffix = f" (min_upvotes≥{self.min_upvotes})" if self.min_upvotes else ""
        logger.info(f"Fetched {len(df)} papers from HF Daily Papers{suffix}")
        return df
