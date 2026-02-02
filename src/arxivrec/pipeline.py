import logging
from abc import ABC, abstractmethod

import pandas as pd

from arxivrec.dataset.fetcher import ArxivFetcher, BaseFetcher
from arxivrec.engine.encoder import TextEncoder
from arxivrec.engine.llm import BaseRanker, OLLMRanker
from arxivrec.notify.notification import BaseNotifier
from arxivrec.topic import Topic

logger = logging.getLogger(__name__)


class EmptyFetchExcpetion(Exception):
    """No fetch result!"""

    pass


class EmailFailExcpetion(Exception):
    """Fail to send email!"""

    pass


class BasePipeline(ABC):
    def __init__(
        self,
        topic: Topic,
        **kwargs,
    ):
        self.topic = topic

    @abstractmethod
    def recommend(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def notify(self):
        pass


class OLLMPipeline(BasePipeline):
    def __init__(
        self,
        topic: Topic | None = None,
        simsearch_top_k: int = 10,
        fetcher: BaseFetcher | None = None,
        encoder: TextEncoder | None = None,
        ollm_ranker: BaseRanker | None = None,
        notifier: BaseNotifier | None = None,
    ):
        super().__init__(topic)
        self.simsearch_top_k = simsearch_top_k
        self.fetcher = fetcher or ArxivFetcher()
        self.encoder = encoder or TextEncoder()
        self.ollm_ranker = ollm_ranker or OLLMRanker()
        self.df_recommendation = None
        self.notifier = notifier

    def __repr__(self):
        return (
            f"{self.__class__.__name__}"
            f"topic='{self.topic}', "
            f"simsearch_top_k={self.simsearch_top_k}, "
            f"fetcher={self.fetcher}(), "
            f"encoder={self.encoder}(), "
            f"ollm_ranker={self.ollm_ranker}(), "
            f"notifier={self.notifier}"
            f")"
        )

    def recommend(self) -> pd.DataFrame:
        df = self.fetcher.fetch()

        if len(df) == 0:
            raise EmptyFetchExcpetion("No items fetched!!")

        logger.info("Start topic and content embedding...")

        my_interest_embedding = self.encoder.encode([self.topic.description])
        content_embedding = self.encoder.encode(df["combined_text"].tolist())
        top_k_indices = self.encoder.get_top_k_similar(
            my_interest_embedding, content_embedding, k=self.simsearch_top_k
        )
        df_simsearch = df.iloc[top_k_indices]

        logger.info("Similarity search done!")

        self.df_recommendation = self.ollm_ranker.rank(
            self.topic.description, df_simsearch
        )

        logger.info(
            f"Recommended articles: {self.df_recommendation.to_json(orient="records")}"
        )

        return self.df_recommendation

    def notify(self):
        EMAIL_TEMPLATE = """
        <html>
        <body>
            <h2>Today's ArXiv Recommendations 🐈</h2>
            <p>Based on your interest: {my_interest}, here are the top 5 papers:</p>
            {input_html_table}
        </body>
        </html>
        """

        try:
            email_columns = ["title", "reasoning", "url"]
            html_table = self.df_recommendation[email_columns].to_html(
                index=False, render_links=True
            )

            email_full_body = EMAIL_TEMPLATE.format(
                my_interest=self.topic.description, input_html_table=html_table
            )
            self.notifier.notify(
                subject="📚 Your Daily ArXiv Digest", body_html=email_full_body
            )
            logger.info("Email notification sent successfully!")

        except Exception:
            raise EmailFailExcpetion("Failed to send email notification!")
