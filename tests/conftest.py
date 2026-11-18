from pathlib import Path

import pytest

from toolharness.data.scenario import read_jsonl

SAMPLE_FILE = Path(__file__).resolve().parents[1] / "data" / "sample" / "hotpotqa_sample.jsonl"


@pytest.fixture
def anyio_backend():
    """Run @pytest.mark.anyio tests on asyncio only."""
    return "asyncio"


@pytest.fixture
def sample_scenarios():
    """The committed HotpotQA sample: 5 questions x 6 fault types."""
    return read_jsonl(SAMPLE_FILE)
