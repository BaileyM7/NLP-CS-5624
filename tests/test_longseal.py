"""Tests for LongSeal -> Scenario conversion. Uses fake rows, no network, no benchmark data."""
import datetime as dt

from toolharness.app.mock_app import MockApp
from toolharness.data.longseal import (
    MAX_PIECE_CHARS,
    MAX_SUPPORTING_FACTS,
    build_scenarios,
    contains_answer,
    is_usable,
    locate_passage,
    parse_date,
    parse_text_fragments,
    row_to_scenarios,
    split_sentences,
    unique_title,
)
from toolharness.data.scenario import FaultType


def make_doc(title, text, date=None):
    return {"title": title, "text": text, "date": date, "url": "https://example.com"}


def make_row(i=0, search_results="conflicting", golds=None, distractors=None, answer="Serban Ghenea", urls=None):
    if golds is None:
        golds = [make_doc("Album of the Year", "History of the award.\nSerban Ghenea has won album of the year five times.", "Apr 03, 2022")]
    if distractors is None:
        distractors = [make_doc("Gossip page", "Unrelated celebrity news."), make_doc(None, "No title here.")]
    return {
        "id": f"ls{i:04d}",
        "question": "Who holds the record for most album of the year wins?",
        "answer": answer,
        "urls": urls or [],
        "golds": golds,
        "12_docs": distractors,
        "search_results": search_results,
        "freshness": "fast-changing",
        "question_types": ["entity/event disambiguation"],
        "topic": "Entertainment",
        "effective_year": "2024",
        "canary": "sealqa:test-canary",
    }


# Parsing helpers

def test_parse_date_formats():
    assert parse_date("Apr 03, 2022") == dt.date(2022, 4, 3)
    assert parse_date("2024-05-01T12:00:00+00:00") == dt.date(2024, 5, 1)
    assert parse_date(None) is None
    assert parse_date("") is None
    assert parse_date("yesterday") is None


def test_split_sentences_keeps_lines_and_breaks_up_long_text():
    assert split_sentences("First line.\n\n  Second line.  ") == ["First line.", "Second line."]
    long_line = " ".join(["This is a sentence."] * 100)  # ~2,000 chars on one line
    pieces = split_sentences(long_line)
    assert len(pieces) == 100
    long_word_run = " ".join(["word"] * 1000)  # one 5,000-char sentence with no punctuation
    for piece in split_sentences(long_word_run):
        assert len(piece) <= MAX_PIECE_CHARS


def test_unique_title_handles_missing_and_repeated_titles():
    seen = set()
    assert unique_title(None, seen) == "Untitled document"
    assert unique_title("Grammys", seen) == "Grammys"
    assert unique_title("grammys", seen) == "grammys (2)"
    assert unique_title("Grammys", seen) == "Grammys (3)"


def test_answer_must_match_a_whole_word():
    assert contains_answer("He was 9 years old.", "9")
    assert not contains_answer("Founded in 1990.", "9")
    assert contains_answer("Won by serban ghenea.", "Serban Ghenea")
    assert contains_answer("Rate was 29.34% that year.", "29.34%")


def test_short_numbers_do_not_match_inside_longer_numbers():
    assert contains_answer("In total 7 states did it.", "7")
    assert contains_answer("The count was 7.", "7")
    assert contains_answer("States (7) qualified.", "7")
    assert not contains_answer("Janata Dal secured 53.7% of votes.", "7")
    assert not contains_answer("A ratio of 7.5 was reported.", "7")
    assert not contains_answer("The city had 1,381 residents.", "381")
    assert not contains_answer("Turnout was 29% overall.", "29")


# Highlighted evidence passages (#:~:text= links)

def test_parse_text_fragments():
    assert parse_text_fragments(["https://a.org/page#:~:text=Serban%20Ghenea%20won"]) == [("Serban Ghenea won", None)]
    with_range = "https://a.org/p#:~:text=pre-,Found%2C%20352,members,-suffix"
    assert parse_text_fragments([with_range]) == [("Found, 352", "members")]
    two = "https://a.org/p#:~:text=first&text=second"
    assert parse_text_fragments([two]) == [("first", None), ("second", None)]
    assert parse_text_fragments(["https://a.org/plain", None, ""]) == []
    assert parse_text_fragments(None) == []


def test_locate_passage_finds_the_covering_sentences():
    sentences = ["Intro text.", "Found 352 airline", "members in total.", "Other text."]
    assert locate_passage(sentences, "FOUND 352", None) == [1]
    assert locate_passage(sentences, "Found 352", "members") == [1, 2]
    assert locate_passage(sentences, "not there", None) == []


def bare_number_row(urls):
    golds = [make_doc("IATA", "IATA history.\nCurrently 381 airlines from 7 regions are members.\nFounded in 1945.[7]")]
    return make_row(golds=golds, answer="7", urls=urls)


def test_bare_number_answers_use_the_evidence_passage():
    row = bare_number_row(["https://iata.org#:~:text=Currently%20381%20airlines"])
    scenario = row_to_scenarios(row)[0]
    assert scenario.metadata["evidence"] == "fragment"
    assert [(f.title, f.sent_id) for f in scenario.supporting_facts] == [("IATA", 1)]


def test_heading_passage_for_a_number_answer_targets_the_rows_below():
    table = "\n".join([
        "Intro paragraph about films.",
        "List of highest-grossing films",
        "| Rank | Title | Year |",
        "| 1 | Avatar | 2009 |",
        "| 2 | Avengers: Endgame | 2019 |",
        "| 3 | Titanic | 1997 |",
        "| 4 | Star Wars | 2015 |",
    ])
    golds = [make_doc("Films", table)]
    row = make_row(golds=golds, answer="10", urls=["https://a.org#:~:text=List%20of%20highest-grossing%20films"])
    facts = row_to_scenarios(row)[0].supporting_facts
    assert [f.sent_id for f in facts] == [3, 4, 5]


def test_short_passage_for_a_text_answer_is_kept():
    golds = [make_doc("Trump", "Personal life\nSpouses * Ivana Zelnickova\nMore text below 1990.")]
    row = make_row(golds=golds, answer="Ivana Trump", urls=["https://a.org#:~:text=Spouses"])
    facts = row_to_scenarios(row)[0].supporting_facts
    assert [f.sent_id for f in facts] == [1]


def test_bare_number_answers_without_a_passage_are_unusable():
    assert not is_usable(bare_number_row(urls=[]))


def test_text_answers_prefer_the_sentence_containing_the_answer():
    urls = ["https://a.org#:~:text=History%20of%20the%20award"]
    scenario = row_to_scenarios(make_row(urls=urls))[0]
    assert scenario.metadata["evidence"] == "answer_match"
    assert [f.sent_id for f in scenario.supporting_facts] == [1]


def test_text_answers_fall_back_to_the_passage():
    urls = ["https://a.org#:~:text=History%20of%20the%20award"]
    scenario = row_to_scenarios(make_row(answer="Taylor Swift", urls=urls))[0]
    assert scenario.metadata["evidence"] == "fragment"
    assert [f.sent_id for f in scenario.supporting_facts] == [0]


# Row -> scenarios

def test_row_expands_to_every_fault_condition():
    scenarios = row_to_scenarios(make_row())
    assert [s.fault_type for s in scenarios] == list(FaultType)
    assert {s.dataset for s in scenarios} == {"longseal"}
    assert scenarios[0].metadata["canary"] == "sealqa:test-canary"
    assert scenarios[0].metadata["search_results"] == "conflicting"


def test_gold_documents_come_first_and_keep_their_dates():
    scenario = row_to_scenarios(make_row())[0]
    assert [doc.title for doc in scenario.context] == ["Album of the Year", "Gossip page", "Untitled document"]
    assert scenario.context[0].last_updated == dt.date(2022, 4, 3)
    assert scenario.context[1].last_updated is None


def test_supporting_facts_only_come_from_gold_documents():
    distractors = [make_doc("Fan page", "Serban Ghenea is great.")]
    scenario = row_to_scenarios(make_row(distractors=distractors))[0]
    assert scenario.fault_target_titles == ["Album of the Year"]
    assert [(f.title, f.sent_id) for f in scenario.supporting_facts] == [("Album of the Year", 1)]


def test_supporting_facts_prefer_sentences_related_to_the_question():
    lines = ["Serban Ghenea was born in Canada."] * 5
    lines.append("Serban Ghenea holds the record for most album of the year wins.")
    golds = [make_doc("Album of the Year", "\n".join(lines))]
    facts = row_to_scenarios(make_row(golds=golds))[0].supporting_facts
    assert len(facts) == MAX_SUPPORTING_FACTS
    assert facts[0].sent_id == 5


def test_rows_without_the_answer_in_a_gold_document_are_unusable():
    assert is_usable(make_row())
    assert not is_usable(make_row(golds=[]))
    assert not is_usable(make_row(answer="Taylor Swift"))


def test_build_keeps_the_search_result_mix():
    rows = [make_row(i, "conflicting" if i % 2 else "unhelpful") for i in range(20)]
    scenarios = build_scenarios(rows, 10, seed=0)
    questions = {s.question_id: s.metadata["search_results"] for s in scenarios}
    assert sorted(questions.values()) == ["conflicting"] * 5 + ["unhelpful"] * 5


# Served by the mock app

def test_long_gold_document_is_paginated_with_the_answer_on_its_page():
    filler = "\n".join(f"Filler sentence number {i} about the history of the award." for i in range(200))
    golds = [make_doc("Album of the Year", filler + "\nSerban Ghenea holds the album of the year record.", "Apr 03, 2022")]
    scenario = row_to_scenarios(make_row(golds=golds))[0]
    app = MockApp(scenario)

    total_pages = len(app.pages["Album of the Year"])
    assert total_pages > 1
    assert app.answer_pages() == {("Album of the Year", total_pages)}
    assert app.get_article("Album of the Year", page=total_pages).last_updated == dt.date(2022, 4, 3)
