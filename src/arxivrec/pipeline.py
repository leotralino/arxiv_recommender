import logging
from abc import ABC, abstractmethod

import pandas as pd

from arxivrec.encoder import TextEncoder
from arxivrec.fetcher import ArxivFetcher, BaseFetcher
from arxivrec.llm import BaseRanker, OLLMRanker

logger = logging.getLogger(__name__)


class EmptyFetchExcpetion(Exception):
    """No fetch result!"""

    pass


class BasePipeline(ABC):
    def __init__(
        self,
        user_interest: str = "I am interested in Retrieval-augmented generation (RAG).",
        **kwargs,
    ):
        self.user_interest = user_interest

    @abstractmethod
    def recommend(self) -> pd.DataFrame:
        pass


class OLLMPipeline(BasePipeline):
    def __init__(
        self,
        user_interest: str = "I am interested in Retrieval-augmented generation (RAG).",
        simsearch_top_k: int = 10,
        fetcher: BaseFetcher | None = None,
        encoder: TextEncoder | None = None,
        ollm_ranker: BaseRanker | None = None,
    ):
        super().__init__(user_interest)
        self.simsearch_top_k = simsearch_top_k
        self.fetcher = fetcher or ArxivFetcher()
        self.encoder = encoder or TextEncoder()
        self.ollm_ranker = ollm_ranker or OLLMRanker()

    def recommend(self) -> pd.DataFrame:
        df = self.fetcher.fetch()

        if len(df) == 0:
            raise EmptyFetchExcpetion("No arxiv articles fetched!!")

        my_interest_embedding = self.encoder.encode([self.user_interest])
        content_embedding = self.encoder.encode(df["combined_text"].tolist())
        top_k_indices = self.encoder.get_top_k_similar(
            my_interest_embedding, content_embedding, k=self.simsearch_top_k
        )
        df_simsearch = df.iloc[top_k_indices]

        df_recommendation = self.ollm_ranker.rank(self.user_interest, df_simsearch)

        return df_recommendation
