"""Tests for generating fact edits (#11). A fake LLM stands in for OpenRouter."""
import json

import pytest
from fakes import fault_variant

from toolharness.app.mock_app import MockApp
from toolharness.data.fact_edits import (
    MAX_ATTEMPTS,
    EditRejected,
    EditReview,
    FactEdit,
    apply_edit,
    build_edits,
    build_prompt,
    missing_edits,
    parse_edit,
    read_edits,
    title_subject,
    write_edits,
)
from toolharness.data.scenario import FaultType

MODEL = "test/model"
SYMPHONY_REPLY = '{"original_value": "Billie Holiday", "new_value": "Ella Fitzgerald", "new_answer": "another film"}'
HOLIDAY_REPLY = '{"original_value": "Lady Day", "new_value": "Lady Swing", "new_answer": "another film"}'


def reply_by_title(prompt):
    """Fake LLM: a valid reply for whichever article the prompt shows."""
    if 'article "Symphony in Black"' in prompt:
        return SYMPHONY_REPLY
    if 'article "Billie Holiday"' in prompt:
        return HOLIDAY_REPLY
    raise AssertionError("unexpected prompt")


def symphony_page(scenario):
    """(sentences, supporting positions) for the Symphony in Black answer page."""
    app = MockApp(scenario)
    return app.get_article("Symphony in Black").sentences, app.supporting_positions("Symphony in Black", 1)


def parse(reply, scenario):
    sentences, _ = symphony_page(scenario)
    return parse_edit(reply, scenario, "Symphony in Black", 1, sentences, MODEL)


# Prompt and reply checks

def test_prompt_shows_the_question_answer_and_marked_sentences(lady_day_scenario):
    sentences, positions = symphony_page(lady_day_scenario)
    prompt = build_prompt(lady_day_scenario, "Symphony in Black", 1, sentences, positions)
    assert lady_day_scenario.question in prompt
    assert "Correct answer: Symphony in Black" in prompt
    assert "* The film, Billie Holiday's screen debut, was directed by Fred Waller." in prompt
    assert "  Symphony in Black is a 1935 musical short featuring Duke Ellington." in prompt


def test_good_reply_is_accepted_even_inside_code_fences(lady_day_scenario):
    edit = parse("```json\n" + SYMPHONY_REPLY + "\n```", lady_day_scenario)
    assert (edit.original_value, edit.new_value, edit.new_answer) == ("Billie Holiday", "Ella Fitzgerald", "another film")
    assert (edit.scenario_id, edit.title, edit.page, edit.model) == (lady_day_scenario.scenario_id, "Symphony in Black", 1, MODEL)


@pytest.mark.parametrize(
    "reply, reason",
    [
        ("I would change the director.", "no JSON object"),
        ("{original_value: Billie}", "not valid JSON"),
        ('{"original_value": "", "new_value": "Ella Fitzgerald", "new_answer": "x"}', "non-empty"),
        ('{"original_value": "1936", "new_value": "1937", "new_answer": "x"}', "does not appear exactly"),
        ('{"original_value": "Billie Holi", "new_value": "Ella", "new_answer": "x"}', "does not appear exactly"),
        ('{"original_value": "Billie Holiday", "new_value": "billie holiday", "new_answer": "x"}', "different"),
        ('{"original_value": "Billie Holiday", "new_value": "Ella", "new_answer": "Symphony in Black"}', "correct answer"),
        ('{"original_value": "Fred Waller", "new_value": "Fred Waller and John Ford", "new_answer": "x"}', "not add"),
        ('{"original_value": "Symphony in Black is", "new_value": "Rhapsody in Blue is", "new_answer": "x"}', "own subject"),
    ],
)
def test_bad_replies_are_rejected_with_a_reason(lady_day_scenario, reply, reason):
    with pytest.raises(EditRejected, match=reason):
        parse(reply, lady_day_scenario)


def test_title_subject_drops_qualifiers():
    assert title_subject("Shin Hyun-joon (actor)") == "Shin Hyun-joon"
    assert title_subject("TD Bank, N.A.") == "TD Bank"
    assert title_subject("Once a Day") == "Once a Day"


def test_fact_in_an_unmarked_sentence_is_accepted(lady_day_scenario):
    edit = parse('{"original_value": "1935", "new_value": "1937", "new_answer": "x"}', lady_day_scenario)
    assert edit.original_value == "1935"


# Applying an edit

def test_edit_replaces_every_whole_mention_only():
    edit = FactEdit(
        scenario_id="s", title="t", page=1, original_value="Holiday", new_value=r"Ella\1", new_answer="x", model=MODEL
    )
    sentences = ["Holiday sang.", "Billie Holiday's debut.", "Holidays are fun."]
    assert apply_edit(sentences, edit) == [r"Ella\1 sang.", r"Billie Ella\1's debut.", "Holidays are fun."]


# Building edits

def test_one_edit_per_answer_page_of_silently_wrong_scenarios_only(lady_day_scenario):
    clean = lady_day_scenario.model_copy(update={"scenario_id": "hotpotqa-ladyday-none", "fault_type": FaultType.NONE})
    edits, failures = build_edits([lady_day_scenario, clean], reply_by_title, MODEL)

    assert failures == []
    assert [(e.title, e.page, e.original_value) for e in edits] == [
        ("Billie Holiday", 1, "Lady Day"),
        ("Symphony in Black", 1, "Billie Holiday"),
    ]
    assert missing_edits([lady_day_scenario, clean], edits) == []


def test_missing_edits_reports_a_shared_page_once_under_the_silently_wrong_id(lady_day_scenario, lady_day_edits):
    stale = fault_variant(lady_day_scenario, FaultType.STALE)
    contradiction = fault_variant(lady_day_scenario, FaultType.CONTRADICTION)
    only_symphony = [e for e in lady_day_edits if e.title == "Symphony in Black"]

    assert missing_edits([lady_day_scenario, stale, contradiction], only_symphony) == [
        "hotpotqa-ladyday-silently_wrong 'Billie Holiday' page 1"
    ]
    assert missing_edits([stale, contradiction], lady_day_edits) == []


def test_missing_edits_needs_only_the_edit_not_the_sibling_scenario(lady_day_scenario, lady_day_edits):
    stale = fault_variant(lady_day_scenario, FaultType.STALE)
    assert missing_edits([stale], lady_day_edits) == []
    assert missing_edits([stale], []) == [
        "hotpotqa-ladyday-silently_wrong 'Billie Holiday' page 1",
        "hotpotqa-ladyday-silently_wrong 'Symphony in Black' page 1",
    ]


def test_rejected_reply_is_retried_with_the_reason(lady_day_scenario):
    prompts = []

    def wrong_first(prompt):
        prompts.append(prompt)
        if len(prompts) == 1:
            return '{"original_value": "1936", "new_value": "1937", "new_answer": "x"}'
        return reply_by_title(prompt)

    edits, failures = build_edits([lady_day_scenario], wrong_first, MODEL)
    assert failures == []
    assert "rejected: '1936' does not appear exactly on this page" in prompts[1]
    assert len(edits) == 2


def test_page_fails_after_max_attempts(lady_day_scenario):
    calls = []

    def always_bad(prompt):
        calls.append(prompt)
        return "no idea"

    edits, failures = build_edits([lady_day_scenario], always_bad, MODEL)
    assert edits == []
    assert len(failures) == 2
    assert len(calls) == 2 * MAX_ATTEMPTS
    assert len(missing_edits([lady_day_scenario], edits)) == 2


def test_existing_edits_are_kept_and_only_missing_pages_are_asked_for(lady_day_scenario):
    first, _ = build_edits([lady_day_scenario], reply_by_title, MODEL)
    asked = []

    def record(prompt):
        asked.append(prompt)
        return reply_by_title(prompt)

    again, failures = build_edits([lady_day_scenario], record, MODEL, existing=first[:1])
    assert failures == []
    assert again == first
    assert len(asked) == 1


# Reviews

def review(scenario, title, action, note):
    return EditReview(scenario_id=scenario.scenario_id, title=title, page=1, action=action, note=note)


def test_skip_reply_is_only_accepted_for_a_reviewed_page(lady_day_scenario):
    sentences, _ = symphony_page(lady_day_scenario)
    reply = '{"skip": "nothing here affects the answer"}'
    with pytest.raises(EditRejected, match="skipping is not allowed"):
        parse(reply, lady_day_scenario)
    edit = parse_edit(reply, lady_day_scenario, "Symphony in Black", 1, sentences, MODEL, allow_skip=True)
    assert edit.skip_reason == "nothing here affects the answer"


def test_skip_review_marks_the_page_without_asking_the_llm(lady_day_scenario):
    first, _ = build_edits([lady_day_scenario], reply_by_title, MODEL)
    skip = review(lady_day_scenario, "Billie Holiday", "skip", "nickname page only")

    def no_calls(prompt):
        raise AssertionError("the LLM should not be asked")

    edits, _ = build_edits([lady_day_scenario], no_calls, MODEL, existing=first, reviews=[skip])
    holiday = next(e for e in edits if e.title == "Billie Holiday")
    assert (holiday.skip_reason, holiday.model) == ("nickname page only", "reviewer")
    assert missing_edits([lady_day_scenario], edits) == []


def test_redo_review_regenerates_once_with_the_note(lady_day_scenario):
    first, _ = build_edits([lady_day_scenario], reply_by_title, MODEL)
    redo = review(lady_day_scenario, "Symphony in Black", "redo", "pick the director instead")
    prompts = []

    def director(prompt):
        prompts.append(prompt)
        return '{"original_value": "Fred Waller", "new_value": "Dudley Murphy", "new_answer": "another film"}'

    edits, _ = build_edits([lady_day_scenario], director, MODEL, existing=first, reviews=[redo])
    symphony = next(e for e in edits if e.title == "Symphony in Black")
    assert (symphony.original_value, symphony.review_note) == ("Fred Waller", "pick the director instead")
    assert "rejected the previous edit for this page ('Billie Holiday' -> 'Ella Fitzgerald'): pick the director" in prompts[0]

    # Running again with the same review keeps the redone edit
    again, _ = build_edits([lady_day_scenario], director, MODEL, existing=edits, reviews=[redo])
    assert again == edits
    assert len(prompts) == 1


def test_edits_round_trip_through_jsonl(tmp_path, lady_day_scenario):
    edits, _ = build_edits([lady_day_scenario], reply_by_title, MODEL)
    path = tmp_path / "edits.jsonl"
    assert write_edits(edits, path) == 2
    assert read_edits([path]) == edits
    assert json.loads(path.read_text(encoding="utf-8").splitlines()[0])["model"] == MODEL
