import logging
from typing import Dict

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter

logger = logging.getLogger(__name__)


class DoclingArxivParser:
    def __init__(self):
        options = PdfPipelineOptions()
        options.do_ocr = False
        options.do_table_structure = False

        self.converter = DocumentConverter(
            format_options={InputFormat.PDF: {"pipeline_options": options}}
        )

    def parse_paper(self, source: str) -> Dict[str, str]:
        """
        Parses an arXiv PDF (via URL or local path) into structured components.
        """
        try:
            result = self.converter.convert(source)
            doc = result.document

            doc_dict = doc.export_to_dict()

            # Extract structured parts based on common LaTeX-style labels
            # Docling labels headers, body text, and lists automatically.
            return {
                "title": doc.name or "Untitled",
                "authors": self._extract_metadata(doc_dict, "authors"),
                "abstract": self._extract_section(doc, "abstract"),
                "main_text": self._extract_main_body(doc),
                "appendix": self._extract_section(doc, "appendix")
                or self._extract_section(doc, "references"),
            }
        except Exception as e:
            logger.error(f"Failed to parse {source}: {e}")
            return {}

    def _extract_section(self, doc, section_name: str) -> str:
        """Helper to find content under a specific heading."""
        text_blocks = []
        found = False
        for item in doc.texts:
            # Simple heuristic: Look for headers containing the section name
            if item.label == "heading" and section_name in item.text.lower():
                found = True
                continue
            # If we've found the header, collect subsequent paragraphs until next header
            if found:
                if item.label == "heading":
                    break
                text_blocks.append(item.text)
        return "\n".join(text_blocks)

    def _extract_main_body(self, doc) -> str:
        """Returns all text items labeled as body or paragraph."""
        return "\n".join(
            [item.text for item in doc.texts if item.label in ["paragraph", "text"]]
        )

    def _extract_metadata(self, doc_dict, key: str) -> str:
        # Docling attempts to extract metadata like authors into a standard field
        metadata = doc_dict.get("metadata", {})
        return metadata.get(key, "Unknown")


if __name__ == "__main__":
    parser = DoclingArxivParser()
    paper_data = parser.parse_paper("https://arxiv.org/pdf/2408.09869.pdf")
    print(paper_data)
