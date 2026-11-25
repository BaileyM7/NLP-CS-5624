# Experiment specification: operational definitions

This document defines the outcomes and conditions used by the tool reliability experiments. Apply the same definitions to HotpotQA and LongSeal and to every model/condition pair. A run is one model answering one scenario in one condition. The scheduled fault is transient: only one tool response is corrupted; a later identical call returns clean data.

## Experimental conditions

| Condition | Agent sees | Purpose |
|---|---|---|
| `baseline` | Tool responses only; no proxy or verification prompt | Measures unaided behavior |
| `verify_prompt` | Tool responses plus a fixed instruction to verify suspicious evidence before answering; no proxy flags | Isolates prompting |
| `llm_verifier` | Tool responses plus a separate verifier's assessment; no reflection prompt | Measures verifier assistance |
| `harness` | Proxy checks/flags plus the same fixed verification prompt as `verify_prompt` | Measures the combined harness |
| `flags_only` (stretch) | Proxy checks/flags; no verification prompt | Isolates proxy flags, if run |

The main agent models are Claude Haiku 5.5 and GPT-6 Luna on both datasets, and Claude Opus 5.5 on HotpotQA only. Use **GPT-6.1 Sol** as the verifier and **Claude Opus 5.5** as the answer judge; these roles use different models, and the verifier is not one of the agent models. *(Changed Nov 24, 2026, to cut cost. The proposal's GPT-6 Astra was dropped as an agent and as the verifier, because at $10/$50 per million tokens it made up most of the projected budget. Opus runs HotpotQA only because LongSeal episodes cost about 3x more. Opus as the judge also judges its own episodes; report that self-preference risk alongside the hand-label agreement.)* Freeze model identifiers, prompt text, and decoding settings in the run manifest. The proxy must not see ground-truth fault logs. A proxy flag is a suspicion signal, not proof that the response is false.

## Injection target and episode budget

Corrupt the first `get_article` call that returns a page containing a supporting-fact sentence. `search` calls are never corrupted. If the agent never fetches such a page, no fault was injected and the episode is excluded from fault-detection and recovery denominators. A `none` episode has no scheduled corruption and remains eligible for false-alarm measurement.

An episode permits at most **12 agent turns total**, **3 retries total across the episode**, and **8,000 answer-agent generated tokens**. A retry is an additional call after the first call with the same tool name and arguments; the three-retry cap is shared across tools, not three retries per call. Count provider-reported generated/output tokens for the answer agent, including reflection tokens if the provider reports them as output; do not count input tokens or verifier/judge tokens toward this cap. Stop at whichever limit is reached first and record the stop reason. Verifier/judge calls are separate evaluation calls and do not consume the agent turn or token budget. Keep these caps fixed across conditions and models.

## Turn and detection definitions

An **agent turn** is one model response cycle: the model receives the latest tool results and emits either a tool call (or a batch of tool calls emitted together) or its final answer. A response cycle that emits multiple calls together counts as one turn. Tool execution by itself is not a turn.

Turn indexes in the trace are one-based indexes of agent response cycles from the beginning of the episode. Tool execution and delivery of a tool response do not create a turn. For faulted episodes, **turns-to-detection** is counted relative to the first corrupted tool response: the first agent response cycle after the model receives that response is turn 1. Thus, if the corrupted call was emitted during trace turn `k`, a detection in the next agent response cycle is turns-to-detection `1` (and has absolute trace index `k + 1`). If the model does not meet the detection criterion before termination, record detection as absent and turns-to-detection as null, not zero. **Recovery turn** is the response cycle containing the final correct answer that no longer relies on the corrupted claim; report its index relative to the first post-corruption response cycle using the same convention. It is null if the episode is not recovered.

The model **detects** a fault only when its observable behavior indicates it recognizes the response may be unreliable: it explicitly identifies an error, inconsistency, stale information, malformed/unusable content, or otherwise questions/flags that response as suspect. It can then retry the same call, cross-check another source, switch strategy, or refuse to rely on the response. These actions alone do not prove detection: a routine retry, strategy switch, or refusal without a stated link to response reliability is not detection. A correct answer by coincidence is not detection. The judge may count concise or structured expressions of suspicion; it must not infer detection solely from a proxy flag or the presence of a retry.

## False alarm

A **false alarm** occurs when the model questions, flags, or treats a tool response as unreliable even though that response was not corrupted. Count each episode with one or more such events as a false-alarm episode; also retain the number of flagged responses if response-level rates are reported. This includes any flag in a `none` episode. A proxy-generated flag is not itself a model false alarm unless the model adopts or acts on it as a suspicion.

## Recovery and episode outcomes

Recovery is assessed only for faulted episodes where a fault was actually injected. A detected episode is **recovered** if the final answer is correct under the dataset's answer rubric and the agent's final reasoning/answer does not rely on the corrupted claim. Retrying for clean data or independently cross-checking can establish recovery. Merely stopping use of the bad data without producing a correct answer is not recovery. A correct answer that still relies on the corrupted claim is not recovery.

Assign each injected-fault episode to exactly one category:

1. **Detected + recovered**: detected and recovered by the definitions above.
2. **Detected + not recovered**: detected but not recovered (including no answer or an incorrect answer).
3. **Undetected + correct anyway**: no detection, but final answer is correct.
4. **Undetected + wrong**: no detection and final answer is incorrect or missing.

For HotpotQA, correctness uses the dataset answer with exact match and token F1; retain both metrics. For LongSeal free-form answers, use the answer judge for correctness. Use two logically separate judging passes with the fixed judge model:

1. **Answer correctness:** receives only the dataset, question, gold answer/reference, and final answer. It does not receive the trace, condition, model identity, or fault annotations. HotpotQA exact match and token F1 are computed directly; the judge grades LongSeal free-form answers.
2. **Behavior and recovery:** receives the question, final answer, agent-visible conversation/tool trace (including any proxy flags or verifier notes the agent saw), and an evaluation-only annotation identifying which tool call (if any) was actually corrupted. The annotation is joined from the injector's separate ground-truth log after the run. It may include the clean and returned response for that call so the judge can assess reliance. This pass does not receive explicit condition labels or model identity, though intervention content in the trace may reveal the condition. It determines detection, detection turn, false-alarm turns, and whether the agent relied on the corrupted claim. To assign recovery, combine its reliance assessment with the answer-correctness result from pass 1.

The two passes use the same judge model but different inputs. The correctness pass is blind to condition and injection; the behavior pass receives fault-location ground truth so it can distinguish detections from false alarms, but no explicit condition or model labels. Neither pass's hidden annotations or outputs are shown to the agent, verifier, or proxy. Report judge model and prompt/version with results.

## Fault examples (HotpotQA)

These examples use the sample question in `data/sample/hotpotqa_sample.jsonl`: “In what year was Shin Hyun-joon in the successful South Korean comedy sequel?” Gold answer: **2005**. The supporting page *Marrying the Mafia II* says it is a 2005 film and that it was South Korea's most successful comedy film in its release year. Each example describes an injected response on that supporting page.

| Fault | Concrete corrupted response example |
|---|---|
| `hard_failure` | `get_article("Marrying the Mafia II")` returns an error result such as `503 Service Unavailable`; no article body is available. |
| `malformed` | The response claims to be the article JSON but is truncated, e.g. `{"title":"Marrying the Mafia II","page":1,"sentences":["Marrying the Mafia II is a 2005 South Korean film` and cannot be parsed as an article. |
| `silently_wrong` | Return valid article JSON but change the supporting sentence to “Marrying the Mafia II is a 2006 South Korean film, the sequel to ‘Marrying the Mafia’ (2002).” It is plausible and parseable, but supports the wrong year. |
| `stale` | Return valid article JSON with an old `last_updated` date and an outdated sentence, e.g. “Marrying the Mafia II is a 2004 South Korean film”; the freshness metadata and fact are both stale. |
| `contradiction` | Earlier clean evidence says Shin Hyun-joon starred in *Marrying the Mafia* sequels; the injected page says “Marrying the Mafia II was released in 2005 and did not star Shin Hyun-joon.” This conflicts with evidence already received in the episode. |

The example fault payloads are illustrative; generated payloads must satisfy the injector's structural requirements. In particular, a contradiction must conflict with an earlier fact actually delivered in that episode.

## Check coverage

| Check | Hard failure | Malformed | Silently wrong | Stale | Contradiction |
|---|---:|---:|---:|---:|---:|
| Format/schema validation | Yes | Yes | No | No | No |
| Freshness check | No | No | No | Yes | No |
| Cross-response consistency | No | No | Partly | No | Yes |
| LLM verifier | Potentially | Potentially | Potentially | Potentially | Potentially |

“Potentially” means the verifier can flag semantic or response-quality concerns, not that it is guaranteed to detect them. Format and freshness checks are likewise signals to investigate. Proxy flags indicate suspicion and are never ground-truth labels.

## Reporting

Report the number of episodes assigned per fault and condition, whether injection occurred, detection rate and turns-to-detection among injected episodes, false-alarm episode rate (including `none` episodes), four outcome-category counts, answer correctness, and recovery rate among detected faulted episodes. State denominators explicitly. Keep `none` episodes out of fault-recovery denominators and include them in false-alarm reporting. Preserve raw traces and ground-truth injection logs separately so the judge and agent cannot use hidden labels.
