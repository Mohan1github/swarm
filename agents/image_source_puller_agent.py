from __future__ import annotations

import logging
import re
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

from .public_web_crawler import (
    fetch_public_url,
    is_public_web_url as _is_public_web_url,
    public_addresses as _public_addresses,
)


MAX_IMAGE_RESULTS = 6
MAX_CRAWL_PAGES = 4
MAX_PAGE_IMAGE_CANDIDATES = 40
MAX_PAGE_BYTES = 1_000_000
LOGGER = logging.getLogger(__name__)
NON_CONTENT_IMAGE_PATTERN = re.compile(
    r"(?:^|[/_.-])(?:logo|icon|avatar|favicon|sprite|tracking|pixel|spacer|"
    r"social-media|facebook|instagram|pinterest|rss|share|copy-link)(?:[/_.-]|$)",
    re.IGNORECASE,
)
IMAGE_REQUEST_PATTERN = re.compile(
    r"\b(?:show|find|search(?:\s+for)?|fetch|pull|source|provide|give|display|look\s+for)\b"
    r".{0,80}\b(?:images?|photos?|pictures?|photographs?|illustrations?|wallpapers?)\b"
    r"|\b(?:images?|photos?|pictures?|photographs?|illustrations?|wallpapers?)\b"
    r".{0,60}\b(?:of|for|show|find|source)\b",
    re.IGNORECASE | re.DOTALL,
)
PRIVATE_QUERY_PATTERN = re.compile(
    r"\b(?:my|our|private|personal|confidential|password|api[\s_-]?key|"
    r"access[\s_-]?token|social[\s_-]?security|ssn)\b"
    r"|\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b"
    r"|\b(?:\+?\d[\d ().-]{8,}\d)\b",
    re.IGNORECASE,
)


class ImageSourceConfigurationError(RuntimeError):
    pass


class _ImageHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.images: list[tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if len(self.images) >= MAX_PAGE_IMAGE_CANDIDATES:
            return
        values = {key.casefold(): value for key, value in attrs if value}
        if tag.casefold() == "img":
            image_url = (
                values.get("src")
                or values.get("data-src")
                or values.get("data-lazy-src")
                or ""
            )
            description = values.get("alt") or values.get("title") or ""
            if image_url:
                self.images.append((image_url, description))
            srcset = values.get("srcset") or values.get("data-srcset") or ""
            if srcset:
                first_source = srcset.split(",", 1)[0].strip().split(" ", 1)[0]
                if first_source:
                    self.images.append((first_source, description))
        elif tag.casefold() == "meta":
            property_name = (
                values.get("property") or values.get("name") or ""
            ).casefold()
            if property_name in {"og:image", "og:image:url", "twitter:image"}:
                image_url = values.get("content") or ""
                if image_url:
                    self.images.append((image_url, ""))


class ImageSourcePullerAgent:
    def __init__(self, search_tool=None, page_fetcher=None) -> None:
        self.search_tool = search_tool
        self.page_fetcher = page_fetcher

    @staticmethod
    def should_search(query: str) -> bool:
        return bool(IMAGE_REQUEST_PATTERN.search(query)) and not bool(
            PRIVATE_QUERY_PATTERN.search(query)
        )

    def _search_pages(self, query: str) -> object:
        if self.search_tool is not None:
            if callable(self.search_tool):
                return self.search_tool(query)
            if hasattr(self.search_tool, "search_pages"):
                return self.search_tool.search_pages(query)
            raise TypeError("Image search tool must be callable or provide search_pages().")

        try:
            from ddgs import DDGS
        except ImportError as error:
            raise ImageSourceConfigurationError(
                "Free image crawling requires the ddgs package."
            ) from error

        return DDGS(timeout=8).text(
            query=query[:2000],
            safesearch="moderate",
            max_results=MAX_CRAWL_PAGES,
            backend="auto",
        )

    def _fetch_page(self, url: str) -> str:
        if self.page_fetcher is not None:
            return self.page_fetcher(url)
        content_type, encoding, body = fetch_public_url(url, MAX_PAGE_BYTES)
        if "text/html" not in content_type:
            return ""
        return body.decode(encoding, errors="replace")

    def retrieve(self, query: str) -> list[dict[str, str]]:
        result = self._search_pages(query)
        if isinstance(result, dict) and result.get("error"):
            raise RuntimeError("The image source search provider returned an error.")
        pages = (
            result.get("results", result.get("pages", []))
            if isinstance(result, dict)
            else result
            if isinstance(result, list)
            else []
        )
        if not isinstance(pages, list):
            return []

        sources = []
        seen_images = set()
        page_count = 0
        fetch_failures = 0
        for page in pages:
            page_url = page if isinstance(page, str) else (
                page.get("href") or page.get("url")
                if isinstance(page, dict)
                else None
            )
            if not _is_public_web_url(page_url):
                continue
            page_count += 1
            if page_count > MAX_CRAWL_PAGES:
                break
            try:
                html = self._fetch_page(page_url)
            except Exception as error:
                fetch_failures += 1
                LOGGER.warning(
                    "Image crawler failed for public host %s (%s)",
                    urlparse(page_url).hostname,
                    type(error).__name__,
                )
                continue

            parser = _ImageHTMLParser()
            parser.feed(html)
            for image_url, description in parser.images:
                absolute_image_url = urljoin(page_url, image_url)
                if (
                    not _is_public_web_url(absolute_image_url)
                    or absolute_image_url in seen_images
                    or NON_CONTENT_IMAGE_PATTERN.search(
                        f"{absolute_image_url} {description}"
                    )
                ):
                    continue
                seen_images.add(absolute_image_url)
                sources.append({
                    "image_url": absolute_image_url,
                    "source_url": page_url,
                    "description": description.strip()[:500],
                })
                if len(sources) >= MAX_IMAGE_RESULTS:
                    return sources

        if not sources and page_count and fetch_failures == page_count:
            raise RuntimeError("The image crawler could not fetch any public result pages.")
        return sources
