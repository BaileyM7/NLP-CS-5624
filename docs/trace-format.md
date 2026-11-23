# Episode trace format

The agent runner writes one JSON object per line, with one file per model and condition. `write_jsonl` and `append_jsonl` enforce that a file contains only one model/condition pair; `trace_filename(model, condition)` supplies the filesystem-safe filename. Use `read_jsonl` to validate and load files.

Each `Episode` identifies the scenario, dataset, model, condition, fault type, question, answer, stop reason, totals, and ordered turns. Conditions are `baseline`, `verify_prompt`, `llm_verifier`, `harness`, and `flags_only`. Each `Turn` represents one model response cycle and contains the emitted assistant text plus zero or more tool calls. Tool calls preserve the exact raw response shown to the agent, parsed arguments, error status, any proxy flags, verifier output, proxy latency, and whether the call repeats an earlier identical call. In the `harness` and `flags_only` conditions, the raw response includes the `_harness` line the model saw, and `harness_flags` lists its flags (`[]` when none).

Only text emitted to the conversation is stored in `assistant_text`; hidden model reasoning is not part of the trace. Tokens and cost are reported per turn and per episode. Verifier usage is kept separate from agent usage, and latency values are milliseconds. Baseline calls have `harness_flags: null`; conditions without verifier output have `verifier_output: null`.

The trace schema forbids undeclared fields and contains no injection ground-truth fields. The injector writes its ground-truth log separately. Evaluation code may join those files later using `episode_id`; never pass the joined labels to the agent.

See [`data/sample/trace_sample.jsonl`](../data/sample/trace_sample.jsonl) for one validated example. The round-trip test is `tests/test_trace.py`.
