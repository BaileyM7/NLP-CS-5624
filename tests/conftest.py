import datetime as dt
import subprocess
import sys
import time
from pathlib import Path

import httpx2
import pytest
from fakes import free_port

from toolharness.data.fact_edits import FactEdit
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
def http_server(tmp_path):
    """Start the tool server on the sample as a subprocess on a free port; yield its base URL.

    Its ground-truth log is tmp_path / "ground_truth.jsonl".
    """
    port = free_port()
    log = open(tmp_path / "server.log", "w")
    process = subprocess.Popen(
        [
            sys.executable, "-m", "toolharness.mcp_server",
            "--scenarios", str(SAMPLE_FILE),
            "--port", str(port),
            "--gt-log", str(tmp_path / "ground_truth.jsonl"),
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    base_url = f"http://127.0.0.1:{port}"

    # Wait until /health answers
    deadline = time.time() + 30
    while True:
        try:
            if httpx2.get(base_url + "/health", timeout=1).status_code == 200:
                break
        except httpx2.HTTPError:
            pass
        if time.time() > deadline or process.poll() is not None:
            process.kill()
            log.close()
            pytest.fail("server did not start:\n" + (tmp_path / "server.log").read_text())
        time.sleep(0.2)

    yield base_url

    process.terminate()
    process.wait(timeout=10)
    log.close()


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


@pytest.fixture
def lady_day_edits(lady_day_scenario):
    """Fact edits for both lady_day answer pages, keyed by its silently_wrong scenario ID."""
    def edit(title, original, new):
        return FactEdit(
            scenario_id=lady_day_scenario.scenario_id,
            title=title,
            page=1,
            original_value=original,
            new_value=new,
            new_answer="another film",
            model="test/model",
        )

    return [edit("Symphony in Black", "Billie Holiday", "Ella Fitzgerald"), edit("Billie Holiday", "Lady Day", "Lady Swing")]
