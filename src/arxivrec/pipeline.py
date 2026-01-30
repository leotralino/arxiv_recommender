import logging
from abc import ABC, abstractmethod

import pandas as pd

from arxivrec.encoder import TextEncoder
from arxivrec.fetcher import fetch_recent_papers
from arxivrec.llm import OLLMRanker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        max_paper_to_fetch: int = 100,
        simsearch_top_k=10,
        encoder=None,
        ollm_ranker=None,
    ):
        super().__init__(user_interest)
        self.max_paper_to_fetch = max_paper_to_fetch
        self.simsearch_top_k = simsearch_top_k
        self.encoder = encoder or TextEncoder()
        self.ollm_ranker = ollm_ranker or OLLMRanker()

    def recommend(self) -> pd.DataFrame:
        df = fetch_recent_papers(max_results=self.max_paper_to_fetch)

        my_interest_embedding = self.encoder.encode([self.user_interest])
        content_embedding = self.encoder.encode(df["combined_text"].tolist())
        top_k_indices = self.encoder.get_top_k_similar(
            my_interest_embedding, content_embedding, k=self.simsearch_top_k
        )
        df_simsearch = df.iloc[top_k_indices]

        df_recommendation = self.ollm_ranker.rank(self.user_interest, df_simsearch)

        return df_recommendation
