"""Mock application: serves one scenario's articles through `search` and `get_article`."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
from typing import Any

from rank_bm25 import BM25Okapi

from ..data.scenario import Document, Scenario
from .backend import (
    AS_OF,
    CLEAN_DATE_WINDOW_DAYS,
    MAX_SEARCH_RESULTS,
    PAGE_CHARS,
    ArticlePage,
    NotFound,
    SearchHit,
    SearchResult,
    ToolResponse,
)

SNIPPET_CHARS = 200
WORD_PATTERN = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does", "for", "from",
    "had", "has", "have", "he", "her", "his", "how", "in", "is", "it", "its", "of", "on",
    "or", "she", "that", "the", "their", "this", "to", "was", "were", "what", "when",
    "where", "which", "who", "whom", "whose", "why", "with",
}


def tokenize(text: str) -> list[str]:
    """Lowercase words with common stopwords removed."""
    words = WORD_PATTERN.findall(text.lower())
    return [word for word in words if word not in STOPWORDS]


def paginate(sentences: list[str], max_chars: int = PAGE_CHARS) -> list[list[int]]:
    """Group sentence indices into pages of at most max_chars characters.

    A sentence longer than max_chars gets a page to itself.
    An article with no sentences still has one empty page.
    """
    pages: list[list[int]] = []
    current: list[int] = []
    current_chars = 0
    for index, sentence in enumerate(sentences):
        size = len(sentence) + 1  # +1 for the space between sentences
        if current and current_chars + size > max_chars:
            pages.append(current)
            current = []
            current_chars = 0
        current.append(index)
        current_chars += size
    if current or not pages:
        pages.append(current)
    return pages


def clean_date(doc: Document) -> dt.date:
    """The article's own date if it has one, else a stable date in the window before AS_OF."""
    if doc.last_updated is not None:
        return doc.last_updated
    digest = hashlib.sha256(doc.title.encode("utf-8")).hexdigest()
    days_back = int(digest, 16) % CLEAN_DATE_WINDOW_DAYS
    return AS_OF - dt.timedelta(days=days_back)


class MockApp:
    """Serves one scenario's articles. Implements ToolBackend."""

    def __init__(self, scenario: Scenario):
        self.scenario = scenario
        self.docs: dict[str, Document] = {}
        self.pages: dict[str, list[list[int]]] = {}
        self.titles_by_lowercase: dict[str, str] = {}
        for doc in scenario.context:
            self.docs[doc.title] = doc
            self.pages[doc.title] = paginate(doc.sentences)
            self.titles_by_lowercase[doc.title.lower()] = doc.title

        # Build the search index: one entry per (title, page), indexed on title + page text
        self.page_keys: list[tuple[str, int]] = []
        corpus: list[list[str]] = []
        for title, pages in self.pages.items():
            for page_number, sentence_ids in enumerate(pages, start=1):
                page_text = " ".join(self.docs[title].sentences[i] for i in sentence_ids)
                self.page_keys.append((title, page_number))
                corpus.append(tokenize(title + " " + page_text))
        self.bm25 = BM25Okapi(corpus)

    def search(self, query: str, k: int = 5) -> SearchResult:
        """Rank article pages against the query with BM25 and return the top k matches."""
        k = max(1, min(k, MAX_SEARCH_RESULTS))
        query_words = tokenize(query)
        scores = self.bm25.get_scores(query_words)

        # Highest score first
        # ties are broken by title, then page num
        order = sorted(range(len(self.page_keys)), key=lambda i: (-scores[i], self.page_keys[i]))

        hits = []
        for i in order:
            if len(hits) == k or scores[i] <= 0:
                break
            title, page = self.page_keys[i]
            hits.append(SearchHit(title=title, page=page, snippet=self._snippet(title, page, query_words)))
        return SearchResult(query=query, hits=hits)

    def get_article(self, title: str, page: int = 1) -> ArticlePage | NotFound:
        """Return one page of an article, or NotFound for an unknown title or page."""
        exact_title = self.titles_by_lowercase.get(title.lower())
        if exact_title is None:
            return NotFound(title=title, message=f"No article titled {title!r}.")

        pages = self.pages[exact_title]
        if page < 1 or page > len(pages):
            return NotFound(
                title=exact_title,
                page=page,
                message=f"Page {page} does not exist; the article has {len(pages)} page(s).",
            )

        doc = self.docs[exact_title]
        sentences = [doc.sentences[i] for i in pages[page - 1]]
        return ArticlePage(
            title=exact_title,
            page=page,
            total_pages=len(pages),
            sentences=sentences,
            last_updated=clean_date(doc),
            source_id=f"{self.scenario.dataset}:{exact_title}",
        )

    def answer_pages(self) -> set[tuple[str, int]]:
        """(title, page) pairs whose page contains a supporting-fact sentence."""
        result: set[tuple[str, int]] = set()
        for fact in self.scenario.supporting_facts:
            pages = self.pages.get(fact.title, [])
            for page_number, sentence_ids in enumerate(pages, start=1):
                if fact.sent_id in sentence_ids:
                    result.add((fact.title, page_number))
        return result

    def call(self, tool: str, args: dict[str, Any]) -> ToolResponse:
        """Run a tool by name and return its result as JSON text."""
        if tool == "search":
            result = self.search(args["query"], args.get("k", 5))
        elif tool == "get_article":
            result = self.get_article(args["title"], args.get("page", 1))
        else:
            return ToolResponse(text=json.dumps({"error": "unknown_tool", "tool": tool}), is_error=True)
        return ToolResponse(text=result.model_dump_json())

    def _snippet(self, title: str, page: int, query_words: list[str]) -> str:
        """The sentence on this page sharing the most words with the query, cut to SNIPPET_CHARS."""
        query_set = set(query_words)
        best_sentence = ""
        best_overlap = -1
        for i in self.pages[title][page - 1]:
            sentence = self.docs[title].sentences[i]
            overlap = len(query_set & set(tokenize(sentence)))
            if overlap > best_overlap:
                best_sentence = sentence
                best_overlap = overlap
        return best_sentence[:SNIPPET_CHARS]
