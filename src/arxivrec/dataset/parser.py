import logging
from typing import Dict

from bs4 import BeautifulSoup
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

logger = logging.getLogger(__name__)


class DoclingArxivParser:
    def __init__(self):
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=PdfPipelineOptions(
                        do_ocr=False, do_table_structure=False
                    )
                )
            }
        )

    def parse_paper(self, source: str) -> Dict[str, str]:
        """
        Parses an arXiv PDF (via URL or local path) into structured components.
        """
        try:
            result = self.converter.convert(source)
            doc = result.document

            doc_html = doc.export_to_html()

            soup = BeautifulSoup(doc_html, "html.parser")
            p_tags = soup.find_all("p")

            # ArXiv specific heuristic:
            # p[0] is usually authors, p[1] is usually affiliation
            affiliation = ""
            if len(p_tags) > 1:
                affiliation = p_tags[1].get_text(strip=True)

            # Main text: join all paragraph tags starting after the affiliation
            # We also skip the first paragraph (authors)
            main_text_chunks = [p.get_text(strip=True) for p in p_tags[2:]]
            main_text = "\n\n".join(main_text_chunks)

            logger.info(
                f"Successfully parsed {source}."
                f"Affiliation detected: {affiliation[:50]}..."
            )

            return {
                "affiliation": affiliation,
                "main_text": main_text,
            }

        except Exception as e:
            logger.error(
                f"Failed to parse {source}: {e}"
                f"Attempting fallback to markdown export..."
            )

            return {"affiliation": "", "main_text": doc.export_to_markdown()}


if __name__ == "__main__":
    from arxivrec.utils.logger import setup_logging

    setup_logging()

    parser = DoclingArxivParser()
    paper_data = parser.parse_paper("https://arxiv.org/pdf/2408.09869.pdf")
    print(paper_data)
