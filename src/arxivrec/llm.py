import json
import logging

import ollama
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _get_authors(author_list: list[str]):
    """
    Formats the author list for LLM input, truncating if too long.
    """
    num_authors = len(author_list)

    if num_authors <= 12:
        author_string = ", ".join(author_list)
    else:
        first_part = author_list[:10]
        last_part = author_list[-2:]
        author_string = f"{', '.join(first_part)} ... {', '.join(last_part)}"

    return author_string


def _parse_llm_output(response_text: str):
    """
    Sometimes LLM does not give list of JSON, so return {"paper": [...]} instead.
    Then parse it accordingly.
    """
    data = json.loads(response_text)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ["papers", "results", "recommendations"]:
            if key in data:
                return data[key]
    return []


def fine_rank_with_llm(top_papers_df: pd.DataFrame, user_interest: str = "RAG"):
    """
    Uses a local LLM to give a qualitative 'Quality Score' to the top candidates.
    """
    top_papers_df = top_papers_df.copy()
    top_papers_df["cleaned_authors"] = top_papers_df["authors"].apply(_get_authors)
    all_papers_json = top_papers_df[["id", "title", "cleaned_authors", "abstract"]].to_dict(orient="records")

    logger.info(f"Number of papers to process: {len(all_papers_json)}")
    logger.info(f"Sample input paper JSON: {all_papers_json[0]}")

    llm_ranking_prompt = f"""
    ## Task

    You are an expert AI and Physics PhD Researcher.
    Given a list of papers with their titles, authors, and abstracts,
    you should help me decide which research papers to read based on my specific interest.
    These papers are posted on arXiv today.
    Your output should be top-5 results, stored in JSON format.

    To select best papers, you should consider relevancy to my interest, novelty, and soundness of methodology.
    Use the author list to gauge credibility if needed.


    ## My Interest
    My research interest topics: "{user_interest}"


    ## Input paper as JSON array
    {all_papers_json}

    ## Output

    In the "papers" field, provide a list of JSON response with top-5 papers, with the following fields:
    - "id": The ArXiv ID of the paper.
    - "reasoning": A one-sentence explanation of why this paper is worth reading or not.

    The output format should look like this:
    {{
        "papers":
        [
            {{"id": "paper_id_1", "reasoning": "Reasoning for paper 1..."}},
            {{"id": "paper_id_2", "reasoning": "Reasoning for paper 2..."}},
            ...
            {{"id": "paper_id_5", "reasoning": "Reasoning for paper 5..."}}
        ]
    }}
    """

    response = ollama.generate(
        model="llama3:8b",
        prompt=llm_ranking_prompt,
        format="json",
        options={"temperature": 0},
    )

    llm_output = _parse_llm_output(response["response"])
    logger.info(f"LLM Output: {llm_output}")

    refined_results = []
    for paper_analysis in llm_output:
        paper_id = paper_analysis["id"]
        matching_paper = top_papers_df[top_papers_df["id"] == paper_id].iloc[0]

        refined_results.append(
            {
                "id": paper_id,
                "url": matching_paper["url"],
                "title": matching_paper["title"],
                "authors": matching_paper["cleaned_authors"],
                "abstract": matching_paper["abstract"],
                "reasoning": paper_analysis["reasoning"],
            }
        )

    return pd.DataFrame(refined_results)
