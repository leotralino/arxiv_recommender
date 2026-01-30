import arxiv
import logging
import pandas as pd
import datetime

# do logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_recent_papers(categories=["cs.LG", "cs.AI", "cs.CV"], days=1, max_results=100):
    """
    Fetches papers from specific categories within a time window.
    """
    # 1. Build Query (e.g., "cat:cs.LG OR cat:cs.AI")
    query = " OR ".join([f"cat:{c}" for c in categories])
    
    client = arxiv.Client(
        page_size=100,
        delay_seconds=3.0, # Be nice to ArXiv servers
        num_retries=3
    )
    
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    
    # 2. Time Window
    threshold = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days)
    
    results = []
    for result in client.results(search):
        # ArXiv results are sorted by date, so we can break early
        if result.published < threshold:
            break
            
        results.append({
            "id": result.entry_id.split('/')[-1],
            "title": result.title,
            "summary": result.summary.replace("\n", " "),
            "url": result.pdf_url,
            "published": result.published,
            "authors": [a.name for a in result.authors],
            "primary_category": result.primary_category
        })
        
    print(f"Fetched {len(results)} new papers since {threshold.strftime('%Y-%m-%d %H:%M')}")
    return pd.DataFrame(results)