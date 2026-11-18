"""Tests for the mock app: pagination, search, get_article, dates, answer pages."""
import datetime as dt
import json
from pathlib import Path

import pytest

from toolharness.app.backend import AS_OF, CLEAN_DATE_WINDOW_DAYS, PAGE_CHARS, ArticlePage, NotFound
from toolharness.app.mock_app import MockApp, clean_date, paginate
from toolharness.data.scenario import Document, FaultType, Scenario, SupportingFact, read_jsonl

SAMPLE_FILE = Path(__file__).resolve().parents[1] / "data" / "sample" / "hotpotqa_sample.jsonl"


def make_scenario(docs: list[Document], facts: list[SupportingFact]) -> Scenario:
    return Scenario(
        scenario_id="test-q1-none",
        question_id="q1",
        dataset="hotpotqa",
        question="Which river flows through Paris?",
        gold_answer="Seine",
        fault_type=FaultType.NONE,
        fault_target_titles=[facts[0].title],
        supporting_facts=facts,
        context=docs,
    )


@pytest.fixture
def small_app() -> MockApp:
    docs = [
        Document(title="Paris", sentences=["Paris is the capital of France.", "The Seine flows through Paris."]),
        Document(title="Berlin", sentences=["Berlin is the capital of Germany.", "The Spree flows through Berlin."]),
        Document(title="Dated Article", sentences=["Some text."], last_updated=dt.date(2006, 3, 1)),
    ]
    return MockApp(make_scenario(docs, [SupportingFact(title="Paris", sent_id=1)]))


# Pagination

def test_pages_stay_under_the_limit_and_keep_every_sentence_once():
    sentences = ["x" * 1000 for _ in range(7)]
    pages = paginate(sentences)
    for page in pages:
        assert sum(len(sentences[i]) + 1 for i in page) <= PAGE_CHARS
    assert [i for page in pages for i in page] == list(range(7))


def test_long_sentence_gets_its_own_page():
    pages = paginate(["short", "y" * 5000, "short"])
    assert pages == [[0], [1], [2]]


def test_empty_article_has_one_empty_page():
    assert paginate([]) == [[]]


# Search

def test_search_ranks_the_matching_article_first(small_app):
    result = small_app.search("Seine river Paris")
    assert result.hits[0].title == "Paris"
    assert result.hits[0].page == 1
    assert result.hits[0].snippet == "The Seine flows through Paris."


def test_search_is_deterministic(small_app):
    assert small_app.search("capital") == small_app.search("capital")


def test_search_drops_pages_with_no_matching_words(small_app):
    assert small_app.search("zebra").hits == []


def test_search_caps_k(small_app):
    assert len(small_app.search("capital", k=100).hits) <= 10


# get_article

def test_get_article_returns_the_page(small_app):
    article = small_app.get_article("Paris")
    assert isinstance(article, ArticlePage)
    assert article.total_pages == 1
    assert article.sentences[1] == "The Seine flows through Paris."
    assert article.source_id == "hotpotqa:Paris"


def test_get_article_ignores_title_case(small_app):
    assert small_app.get_article("paris").title == "Paris"


def test_unknown_title_or_page_is_not_found(small_app):
    assert isinstance(small_app.get_article("Rome"), NotFound)
    missing_page = small_app.get_article("Paris", page=2)
    assert isinstance(missing_page, NotFound)
    assert missing_page.page == 2


# Dates

def test_undated_articles_get_a_stable_date_inside_the_window():
    doc = Document(title="Paris", sentences=["text"])
    date = clean_date(doc)
    assert date == clean_date(doc)
    assert AS_OF - dt.timedelta(days=CLEAN_DATE_WINDOW_DAYS) < date <= AS_OF


def test_dated_articles_keep_their_date(small_app):
    assert small_app.get_article("Dated Article").last_updated == dt.date(2006, 3, 1)


# Answer pages

def test_answer_pages_point_at_the_page_holding_the_fact():
    long_doc = Document(title="Long", sentences=["z" * 2000, "w" * 2000, "The answer is here."])
    app = MockApp(make_scenario([long_doc], [SupportingFact(title="Long", sent_id=2)]))
    assert app.answer_pages() == {("Long", 2)}


# call()

def test_call_returns_json_text(small_app):
    response = small_app.call("get_article", {"title": "Paris", "page": 1})
    assert not response.is_error
    assert json.loads(response.text)["title"] == "Paris"


def test_unknown_tool_is_an_error(small_app):
    assert small_app.call("delete_everything", {}).is_error


# Real HotpotQA sample

@pytest.mark.skipif(not SAMPLE_FILE.exists(), reason="sample not generated yet")
def test_hotpotqa_articles_fit_on_one_page():
    for scenario in read_jsonl(SAMPLE_FILE):
        app = MockApp(scenario)
        for title in app.pages:
            assert len(app.pages[title]) == 1


@pytest.mark.skipif(not SAMPLE_FILE.exists(), reason="sample not generated yet")
def test_search_finds_a_supporting_article_for_every_sample_question():
    for scenario in read_jsonl(SAMPLE_FILE):
        if scenario.fault_type != FaultType.NONE:
            continue
        hit_titles = [hit.title for hit in MockApp(scenario).search(scenario.question).hits]
        assert set(hit_titles) & set(scenario.fault_target_titles), scenario.question
