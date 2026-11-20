# toolharness

A shared Python project for building and evaluating tool‑using agents.

## Setup

```bash
# Clone the repository
git clone https://github.com/BaileyM7/NLP-CS-5624.git
cd NLP-CS-5624

# Create a virtual environment (Technically optional)
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate

# Install the package in editable mode with development extras
pip install -e .[dev]
```

## Configuration

Copy the example environment file and fill in your OpenRouter API key:

```bash
cp .env.example .env
# edit .env and set OPENROUTER_API_KEY=your_key_here
```

## Running tests

```bash
pytest
```

The placeholder test should pass, confirming the package installs correctly.

## Experiment documentation

- [Experiment specification](docs/experiment-spec.md): conditions, fault definitions, outcome metrics, and episode budgets.

## Project layout

```
src/toolharness/
    data/      # dataset loading + Scenario model (HotpotQA, LongSeal)
    app/       # mock application: the "real" tools with ground truth
    mcp_server/ # MCP server exposing the tools
    injector/  # fault injector: corrupts scheduled tool responses
    proxy/     # proxy utilities
    agent/     # agent orchestration
    eval/      # evaluation utilities
tests/        # test suite
scripts/      # helper scripts
docs/         # documentation
```

Feel free to extend the package with actual functionality.
