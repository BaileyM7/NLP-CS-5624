"""Tests for the malformed-output fault (#10)."""
import json
import random

import pytest
from pydantic import ValidationError

from toolharness.app.backend import ArticlePage
from toolharness.app.mock_app import MockApp
from toolharness.data.scenario import FaultType
from toolharness.injector.corruptors import AnswerTarget
from toolharness.injector.faults import CORRUPTORS
from toolharness.injector.faults.malformed import HTML_ERROR_PAGE, VARIANTS, break_page
from toolharness.injector.ground_truth import GroundTruthLog
from toolharness.injector.injector import FaultInjector


@pytest.fixture
def scenarios(sample_scenarios):
    return [s for s in sample_scenarios if s.fault_type == FaultType.MALFORMED]


def answer_page(scenario):
    """The scenario's first answer page: (app, get_article args, clean text, target)."""
    app = MockApp(scenario)
    title, page = sorted(app.answer_pages())[0]
    args = {"title": title, "page": page}
    target = AnswerTarget(title, page, app.supporting_positions(title, page), scenario.gold_answer)
    return app, args, app.call("get_article", args).text, target


def variant_of(text):
    """Name the malformed variant from the broken text."""
    if text == HTML_ERROR_PAGE:
        return "invalid_json"
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return "truncated"
    return "missing_field" if "sentences" not in data else "wrong_type"


def test_every_variant_fails_the_page_schema(scenarios):
    for scenario in scenarios:
        _, _, clean, target = answer_page(scenario)
        for variant in VARIANTS:
            text = break_page(variant, clean, target, random.Random(0))
            with pytest.raises(ValidationError):
                ArticlePage.model_validate_json(text)


def test_no_variant_keeps_a_full_supporting_sentence(scenarios):
    for scenario in scenarios:
        _, _, clean, target = answer_page(scenario)
        supporting = [json.loads(clean)["sentences"][i] for i in target.sentence_positions]
        for variant in VARIANTS:
            for seed in range(5):
                text = break_page(variant, clean, target, random.Random(seed))
                for sentence in supporting:
                    assert json.dumps(sentence, ensure_ascii=False) not in text


def test_truncated_response_is_a_prefix_of_the_clean_one(scenarios):
    for scenario in scenarios:
        _, _, clean, target = answer_page(scenario)
        for seed in range(5):
            assert clean.startswith(break_page("truncated", clean, target, random.Random(seed)))


def test_truncation_cuts_inside_the_supporting_sentence_before_the_answer():
    page = {
        "title": "Marrying the Mafia II",
        "page": 1,
        "total_pages": 1,
        "sentences": ["A 2005 comedy film.", "It stars Shin Hyun-joon and Kim Won-hee.", "It was a hit."],
        "last_updated": "2026-09-14",
        "source_id": "src-1",
    }
    clean = json.dumps(page, separators=(",", ":"))
    target = AnswerTarget("Marrying the Mafia II", 1, [1], "Shin Hyun-joon")
    prefix = '{"title":"Marrying the Mafia II","page":1,"total_pages":1,"sentences":["A 2005 comedy film.","'

    for seed in range(20):
        text = break_page("truncated", clean, target, random.Random(seed))
        assert text.startswith(prefix)
        kept = text[len(prefix):]
        assert "It stars ".startswith(kept)  # only text before the answer survives


def test_variant_is_picked_by_seed_and_scenario(scenarios):
    def corrupted(scenario, seed, episode_id):
        app, args, _, _ = answer_page(scenario)
        injector = FaultInjector(app, scenario, CORRUPTORS[FaultType.MALFORMED], seed, episode_id)
        return injector.call("get_article", args)

    assert corrupted(scenarios[0], 0, "opus-ep") == corrupted(scenarios[0], 0, "haiku-ep")
    seen = {variant_of(corrupted(scenarios[0], seed, "ep").text) for seed in range(40)}
    assert seen == set(VARIANTS)


def test_broken_page_is_a_normal_result_and_a_retry_is_clean(tmp_path, scenarios):
    scenario = scenarios[0]
    app, args, clean, _ = answer_page(scenario)
    log_path = tmp_path / "gt.jsonl"
    injector = FaultInjector(app, scenario, CORRUPTORS[FaultType.MALFORMED], 0, "ep-1", GroundTruthLog(log_path))

    broken = injector.call("get_article", args)
    retry = injector.call("get_article", args)

    assert not broken.is_error
    assert broken.text != clean
    assert retry.text == clean
    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["corrupted"] is True
    assert rows[0]["corrupted_is_error"] is False
    assert rows[1]["retry_of_corrupted"] is True
