import json

from toolharness.eval.trace import (
    Episode,
    append_jsonl,
    read_jsonl,
    trace_filename,
    write_jsonl,
)


def sample_episode() -> Episode:
    return Episode.model_validate(
        {
            "episode_id": "sample-hotpotqa-baseline-001",
            "scenario_id": "hotpotqa-5ab797af5542995dae37e96c-none",
            "dataset": "hotpotqa",
            "model": "sample-model",
            "condition": "baseline",
            "fault_type": "none",
            "question": "In what year was Shin Hyun-joon in the successful South Korean comedy sequel?",
            "final_answer": "2005",
            "gold_answer": "2005",
            "stop_reason": "final_answer",
            "totals": {
                "agent_tokens": {"input_tokens": 42, "output_tokens": 18, "total_tokens": 60},
                "agent_cost_usd": 0.00012,
                "latency_ms": 812.5,
            },
            "turns": [
                {
                    "index": 1,
                    "assistant_text": "I should inspect the supporting article.",
                    "tool_calls": [
                        {
                            "tool_name": "get_article",
                            "tool_args": {"title": "Marrying the Mafia II", "page": 1},
                            "raw_tool_response": '{"title":"Marrying the Mafia II","sentences":["A 2005 South Korean film."]}',
                            "harness_flags": None,
                        }
                    ],
                    "totals": {
                        "tokens": {"input_tokens": 42, "output_tokens": 18, "total_tokens": 60},
                        "cost_usd": 0.00012,
                        "latency_ms": 812.5,
                    },
                }
            ],
        }
    )


def test_episode_jsonl_round_trip(tmp_path):
    episode = sample_episode()
    path = tmp_path / trace_filename(episode.model, episode.condition)

    assert write_jsonl([episode], path) == 1
    loaded = read_jsonl(path)

    assert len(loaded) == 1
    assert loaded[0] == episode
    assert "ground_truth" not in json.loads(path.read_text(encoding="utf-8"))


def test_jsonl_file_rejects_mixed_model_condition_pairs(tmp_path):
    first = sample_episode()
    second = first.model_copy(update={"condition": "harness"})
    path = tmp_path / "mixed.jsonl"

    try:
        write_jsonl([first, second], path)
    except ValueError as error:
        assert "one model/condition pair" in str(error)
    else:
        raise AssertionError("mixed model/condition files must be rejected")
    assert not path.exists()


def test_append_jsonl_rejects_different_partition(tmp_path):
    first = sample_episode()
    second = first.model_copy(update={"model": "different-model"})
    path = tmp_path / trace_filename(first.model, first.condition)
    write_jsonl([first], path)

    try:
        append_jsonl(second, path)
    except ValueError as error:
        assert "different model/condition" in str(error)
    else:
        raise AssertionError("append must preserve the existing partition")
    assert read_jsonl(path) == [first]


def test_committed_sample_validates():
    from pathlib import Path

    sample_path = Path(__file__).parents[1] / "data" / "sample" / "trace_sample.jsonl"
    sample = read_jsonl(sample_path)

    assert len(sample) == 1
    assert sample[0].episode_id == "sample-hotpotqa-baseline-001"


def test_jsonl_reports_invalid_line_number(tmp_path):
    path = tmp_path / "broken.jsonl"
    path.write_text(sample_episode().model_dump_json() + "\nnot-json\n", encoding="utf-8")

    try:
        read_jsonl(path)
    except ValueError as error:
        assert "line 2" in str(error)
    else:
        raise AssertionError("invalid JSONL should fail validation")


def test_trace_rejects_ground_truth_fields():
    data = sample_episode().model_dump()
    data["fault_injected"] = True

    try:
        Episode.model_validate(data)
    except ValueError as error:
        assert "fault_injected" in str(error)
    else:
        raise AssertionError("ground-truth fields must not enter agent traces")
