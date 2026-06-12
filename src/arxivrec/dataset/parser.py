import io

import pdfplumber
import requests


def get_header_text(url: str) -> str:
    """Extract first-page text from an ArXiv PDF for affiliation matching."""
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    with pdfplumber.open(io.BytesIO(response.content)) as pdf:
        if not pdf.pages:
            return ""
        return pdf.pages[0].extract_text() or ""


if __name__ == "__main__":
    text = get_header_text("https://arxiv.org/pdf/2408.09869.pdf")
    print(text[:500])
