import datetime as dt
from pathlib import Path

import pytest

from toolharness.data.scenario import Document, FaultType, Scenario, SupportingFact, read_jsonl

SAMPLE_FILE = Path(__file__).resolve().parents[1] / "data" / "sample" / "hotpotqa_sample.jsonl"


@pytest.fixture
def anyio_backend():
    """Run @pytest.mark.anyio tests on asyncio only."""
    return "asyncio"


@pytest.fixture
def sample_scenarios():
    """The committed HotpotQA sample: 5 questions x 6 fault types."""
    return read_jsonl(SAMPLE_FILE)


@pytest.fixture
def lady_day_scenario():
    """A small two-hop silently_wrong scenario with two answer pages."""
    return Scenario(
        scenario_id="hotpotqa-ladyday-silently_wrong",
        question_id="ladyday",
        dataset="hotpotqa",
        question='What film was the screen debut of the musician nicknamed "Lady Day"?',
        gold_answer="Symphony in Black",
        fault_type=FaultType.SILENTLY_WRONG,
        fault_target_titles=["Billie Holiday", "Symphony in Black"],
        supporting_facts=[
            SupportingFact(title="Billie Holiday", sent_id=1),
            SupportingFact(title="Symphony in Black", sent_id=1),
        ],
        context=[
            Document(
                title="Symphony in Black",
                sentences=[
                    "Symphony in Black is a 1935 musical short featuring Duke Ellington.",
                    "The film, Billie Holiday's screen debut, was directed by Fred Waller.",
                    "Billie Holiday sang in one scene.",
                ],
                last_updated=dt.date(2026, 9, 1),
            ),
            Document(
                title="Billie Holiday",
                sentences=[
                    "Billie Holiday was an American jazz singer.",
                    'Nicknamed "Lady Day" by Lester Young, Holiday shaped jazz singing.',
                ],
            ),
            Document(title="Duke Ellington", sentences=["Duke Ellington was an American composer."]),
        ],
    )
