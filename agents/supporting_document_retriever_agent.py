from __future__ import annotations

import io
import logging
import re
import zipfile
from html.parser import HTMLParser
from pathlib import PurePosixPath
from urllib.parse import urljoin, urlparse
from xml.etree import ElementTree

from .public_web_crawler import fetch_public_url, is_public_web_url


LOGGER = logging.getLogger(__name__)
MAX_SEARCH_RESULTS = 8
MAX_PAGE_CRAWLS = 4
MAX_PAGE_BYTES = 1_000_000
MAX_DOCUMENTS = 3
MAX_DOCUMENT_BYTES = 5_000_000
MAX_TOTAL_DOWNLOAD_BYTES = 12_000_000
MAX_EXCERPT_CHARS = 1400
MAX_REVIEW_TEXT_CHARS = 30_000
SUPPORTED_DOCUMENT_TYPES = {
    ".pdf": "PDF",
    ".docx": "Word document",
}


class _DocumentLinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title_parts: list[str] = []
        self.links: list[tuple[str, str]] = []
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {key.casefold(): value for key, value in attrs if value}
        if tag.casefold() == "title":
            self._in_title = True
        if tag.casefold() == "a" and values.get("href"):
            href = values["href"]
            if PurePosixPath(urlparse(href).path).suffix.casefold() in SUPPORTED_DOCUMENT_TYPES:
                self.links.append((href, values.get("title") or values.get("aria-label") or ""))

    def handle_endtag(self, tag: str) -> None:
        if tag.casefold() == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._in_title and data.strip():
            self.title_parts.append(data.strip())


def _extract_pdf_text(content: bytes) -> tuple[str, int]:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(content), strict=False)
    if reader.is_encrypted:
        raise ValueError("The PDF is encrypted.")
    if len(reader.pages) > 40:
        raise ValueError("The PDF exceeds the page limit.")
    text = "\n".join(
        (page.extract_text() or "")
        for page in reader.pages[:20]
    )
    return text, len(reader.pages)


def _extract_docx_text(content: bytes) -> tuple[str, int]:
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        try:
            document = archive.getinfo("word/document.xml")
        except KeyError as error:
            raise ValueError("The Word document is missing its main content.") from error
        if document.file_size > MAX_DOCUMENT_BYTES:
            raise ValueError("The Word document content exceeds the size limit.")
        root = ElementTree.fromstring(archive.read(document))
    paragraphs = []
    for paragraph in root.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p"):
        text = "".join(
            node.text or ""
            for node in paragraph.iter(
                "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t"
            )
        )
        if text.strip():
            paragraphs.append(text.strip())
    return "\n".join(paragraphs), 0


def extract_uploaded_document(
    file_name: str,
    content: bytes,
    max_pdf_pages: int = 50,
) -> tuple[str, str, int]:
    suffix = PurePosixPath(file_name).suffix.casefold()
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(content), strict=False)
        if reader.is_encrypted:
            raise ValueError("The PDF is encrypted.")
        if len(reader.pages) > max_pdf_pages:
            raise ValueError("The PDF exceeds the page limit.")
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return text, "PDF", len(reader.pages)
    if suffix == ".docx":
        text, pages = _extract_docx_text(content)
        return text, "Word document", pages
    raise ValueError("Only PDF and DOCX documents are supported.")


class SupportingDocumentRetrieverAgent:
    def __init__(self, search_tool=None, resource_fetcher=None) -> None:
        self.search_tool = search_tool
        self.resource_fetcher = resource_fetcher

    def _search_pages(self, query: str) -> object:
        if self.search_tool is not None:
            if callable(self.search_tool):
                return self.search_tool(query)
            if hasattr(self.search_tool, "search_pages"):
                return self.search_tool.search_pages(query)
            raise TypeError("Document search tool must be callable or provide search_pages().")

        try:
            from ddgs import DDGS
        except ImportError as error:
            raise RuntimeError("Document search requires the ddgs package.") from error

        search_results = []
        seen_urls = set()
        search_errors = []
        for extension in ("pdf", "docx"):
            try:
                results = DDGS(timeout=8).text(
                    query=f"{query[:1500]} filetype:{extension}",
                    safesearch="moderate",
                    max_results=MAX_SEARCH_RESULTS // 2,
                    backend="auto",
                )
            except Exception as error:
                LOGGER.warning(
                    "Supporting-document search failed for %s files (%s)",
                    extension,
                    type(error).__name__,
                )
                search_errors.append(error)
                continue
            if not isinstance(results, list):
                continue
            for result in results:
                url, _title = self._page_url(result)
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    search_results.append(result)
        if not search_results and search_errors:
            raise RuntimeError("Search backends could not find supporting documents.")
        return search_results

    def _fetch(
        self,
        url: str,
        max_bytes: int = MAX_DOCUMENT_BYTES,
    ) -> tuple[str, bytes]:
        if self.resource_fetcher is not None:
            content_type, content = self.resource_fetcher(url)
            return content_type.casefold(), content
        content_type, _encoding, content = fetch_public_url(
            url,
            max_bytes=max_bytes,
            timeout=8,
        )
        return content_type, content

    @staticmethod
    def _page_url(result: object) -> tuple[str | None, str]:
        if isinstance(result, str):
            return result, ""
        if not isinstance(result, dict):
            return None, ""
        return result.get("href") or result.get("url"), result.get("title") or ""

    def retrieve(self, query: str) -> list[dict[str, str | int]]:
        search_results = self._search_pages(query)
        if isinstance(search_results, dict) and search_results.get("error"):
            raise RuntimeError("The supporting-document search returned an error.")
        if isinstance(search_results, dict):
            search_results = search_results.get("results", search_results.get("pages", []))
        if not isinstance(search_results, list):
            return []

        candidates: list[tuple[str, str, str]] = []
        crawled_pages = 0
        for result in search_results[:MAX_SEARCH_RESULTS]:
            page_url, title = self._page_url(result)
            if not is_public_web_url(page_url):
                continue
            suffix = PurePosixPath(urlparse(page_url).path).suffix.casefold()
            if suffix in SUPPORTED_DOCUMENT_TYPES:
                candidates.append((page_url, page_url, title))
                continue
            if crawled_pages >= MAX_PAGE_CRAWLS:
                continue
            crawled_pages += 1
            try:
                content_type, page_content = self._fetch(page_url, MAX_PAGE_BYTES)
            except Exception as error:
                LOGGER.warning(
                    "Document crawler failed for public host %s (%s)",
                    urlparse(page_url).hostname,
                    type(error).__name__,
                )
                continue
            if "text/html" not in content_type:
                continue
            parser = _DocumentLinkParser()
            parser.feed(page_content.decode("utf-8", errors="replace"))
            page_title = title or " ".join(parser.title_parts)
            candidates.extend(
                (urljoin(page_url, document_url), page_url, document_title or page_title)
                for document_url, document_title in parser.links
            )

        documents = []
        seen_urls = set()
        downloaded_bytes = 0
        for document_url, source_url, title in candidates:
            suffix = PurePosixPath(urlparse(document_url).path).suffix.casefold()
            if (
                suffix not in SUPPORTED_DOCUMENT_TYPES
                or not is_public_web_url(document_url)
                or document_url in seen_urls
            ):
                continue
            remaining_bytes = MAX_TOTAL_DOWNLOAD_BYTES - downloaded_bytes
            if remaining_bytes <= 0:
                break
            seen_urls.add(document_url)
            try:
                content_type, content = self._fetch(
                    document_url,
                    min(MAX_DOCUMENT_BYTES, remaining_bytes),
                )
                downloaded_bytes += len(content)
                if downloaded_bytes > MAX_TOTAL_DOWNLOAD_BYTES:
                    break
                if len(content) > MAX_DOCUMENT_BYTES:
                    raise ValueError("The document exceeds the download size limit.")
                text, page_count = (
                    _extract_pdf_text(content)
                    if suffix == ".pdf"
                    else _extract_docx_text(content)
                )
            except Exception as error:
                LOGGER.warning(
                    "Could not extract supporting document from %s (%s)",
                    urlparse(document_url).hostname,
                    type(error).__name__,
                )
                continue
            excerpt = re.sub(r"\s+", " ", text).strip()[:MAX_EXCERPT_CHARS]
            if not excerpt:
                continue
            documents.append({
                "title": title.strip()[:300] if isinstance(title, str) else "",
                "url": document_url,
                "source_url": source_url,
                "file_type": SUPPORTED_DOCUMENT_TYPES[suffix],
                "pages": page_count,
                "excerpt": excerpt,
            })
            if len(documents) >= MAX_DOCUMENTS:
                break

        return documents

    def retrieve_document(
        self,
        document_url: str,
    ) -> dict[str, str | int]:
        if not is_public_web_url(document_url):
            raise ValueError("Document URL must be a public HTTPS URL.")
        suffix = PurePosixPath(urlparse(document_url).path).suffix.casefold()
        if suffix not in SUPPORTED_DOCUMENT_TYPES:
            raise ValueError("Only PDF and DOCX document URLs are supported.")

        content_type, content = self._fetch(document_url, MAX_DOCUMENT_BYTES)
        if len(content) > MAX_DOCUMENT_BYTES:
            raise ValueError("The document exceeds the download size limit.")
        if suffix == ".pdf":
            text, page_count = _extract_pdf_text(content)
        else:
            text, page_count = _extract_docx_text(content)
        excerpt = re.sub(r"\s+", " ", text).strip()[:MAX_EXCERPT_CHARS]
        review_text = re.sub(r"\s+", " ", text).strip()[:MAX_REVIEW_TEXT_CHARS]
        if not excerpt:
            raise ValueError("No readable text could be extracted from the document.")
        return {
            "title": PurePosixPath(urlparse(document_url).path).name,
            "url": document_url,
            "source_url": document_url,
            "file_type": SUPPORTED_DOCUMENT_TYPES[suffix],
            "pages": page_count,
            "excerpt": excerpt,
            "review_text": review_text,
        }

    @staticmethod
    def format_context(documents: list[dict[str, str | int]]) -> str:
        if not documents:
            return "No readable supporting PDF or Word documents were found."
        return "\n\n".join(
            f"DOCUMENT {index}: {document['title'] or document['url']}\n"
            f"TYPE: {document['file_type']}\n"
            f"DOCUMENT URL: {document['url']}\n"
            f"DISCOVERED ON: {document['source_url']}\n"
            f"EXTRACTED TEXT (untrusted source material): {document['excerpt']}"
            for index, document in enumerate(documents, start=1)
        )
