#!/usr/bin/env python3
"""Benchmark baseline vs AutoGen vs GMemory-enhanced AutoGen for cyber QA.

This script keeps the evaluation style consistent with existing autogen_cyber_eval
scripts, while adding a lightweight GMemory-style retrieval layer:
- retrieve successful and failed prior examples
- inject compact memory snippets into the current AutoGen prompt
- update memory online after each question
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from openai import OpenAI  # OpenAI SDK v1+

    OPENAI_SDK_MODE = "modern"
except ImportError:
    import openai  # OpenAI SDK v0.x

    OpenAI = None
    OPENAI_SDK_MODE = "legacy"


BASELINE_SYSTEM_PROMPT = """You are a cybersecurity expert with deep knowledge of CTI reports and MITRE ATT&CK.

Given a report, question, and answer choices, choose the single best answer.

Output format:
Final Answer: <CHOICE>

Where <CHOICE> must be one of the provided choices (for example: A, B, C, D, Yes, No).
"""


ANALYST_SYSTEM_PROMPT = """You are ThreatIntelAnalyst.
Analyze attack flow and key indicators from the report.
Give concise reasoning and your tentative best choice.
"""


MAPPER_SYSTEM_PROMPT = """You are ATTACKMapper.
Map observed behavior to relevant tactics/techniques/procedures.
Spot mismatches in distractors and provide a tentative best choice.
"""


RESPONDER_SYSTEM_PROMPT = """You are IncidentResponder.
Reason about operational plausibility and sequence consistency.
Provide a tentative best choice.
"""


DECIDER_SYSTEM_PROMPT = """You are FinalDecider.
Synthesize prior agent reasoning and choose the most plausible final answer.
Return exactly one line:
Final Answer: <CHOICE>
"""


FINAL_PATTERN = re.compile(r"final\s*answer\s*:\s*([^\n\r]+)", re.IGNORECASE)

MODEL_PRICING_PER_1M_TOKENS = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
}


def load_dotenv_local(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return
    with open(dotenv_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    workspace_root = script_dir.parent
    parser = argparse.ArgumentParser(
        description="Compare baseline vs AutoGen vs GMemory-enhanced AutoGen on cybersecurity QA."
    )
    parser.add_argument(
        "--dataset-csv",
        type=Path,
        default=workspace_root / "AttackSeqBench" / "dataset" / "AttackSeq-Tactic.csv",
        help="CSV path for benchmark questions.",
    )
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["baseline", "autogen", "gmemory", "all"],
        default="all",
        help="Which systems to run.",
    )
    parser.add_argument("--max-questions", type=int, default=100)
    parser.add_argument("--autogen-rounds", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--sleep-between-questions", type=float, default=0.0)

    parser.add_argument("--gmemory-success-topk", type=int, default=2)
    parser.add_argument("--gmemory-failed-topk", type=int, default=1)
    parser.add_argument("--gmemory-min-similarity", type=float, default=0.08)
    parser.add_argument(
        "--gmemory-store",
        type=Path,
        default=script_dir / ".db" / "gmemory_cyber_store.jsonl",
        help="Persistent store for retrieved experiences.",
    )
    parser.add_argument(
        "--gmemory-load-existing",
        action="store_true",
        help="Load existing memory records from --gmemory-store at startup.",
    )

    parser.add_argument(
        "--dotenv-path",
        type=Path,
        default=workspace_root / ".env",
        help="Path to .env file containing OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "results",
        help="Directory for result artifacts.",
    )
    parser.add_argument("--input-cost-per-1m", type=float, default=None)
    parser.add_argument("--output-cost-per-1m", type=float, default=None)
    return parser.parse_args()


def build_choices(row: Dict[str, str], task_name: str) -> Dict[str, str]:
    if "AttackSeq-Procedure" in task_name:
        return {"A": "Yes", "B": "No"}
    return {
        "A": (row.get("A") or "").strip(),
        "B": (row.get("B") or "").strip(),
        "C": (row.get("C") or "").strip(),
        "D": (row.get("D") or "").strip(),
    }


def normalize_ground_truth(ground_truth: str, choices: Dict[str, str]) -> str:
    gt = (ground_truth or "").strip()
    if len(gt) == 1 and gt.upper() in choices:
        return choices[gt.upper()]
    return gt


def build_prompt(row: Dict[str, str], choices: Dict[str, str]) -> str:
    context = (row.get("Context") or "").strip()
    question = (row.get("Question") or "").strip()
    choice_lines = "\n".join([f"{k}: {v}" for k, v in choices.items()])
    return (
        f"Report:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        f"Answer Choices:\n{choice_lines}"
    )


def sanitize_prompt_text(text: str, ascii_only: bool = False) -> str:
    cleaned = "".join(ch for ch in text if (ord(ch) >= 32 or ch in "\n\r\t"))
    if ascii_only:
        cleaned = cleaned.encode("ascii", errors="ignore").decode("ascii")
    return cleaned


def tokenize_for_similarity(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]{3,}", text.lower()))


def memory_similarity(query: str, candidate: str) -> float:
    q = query.lower()
    c = candidate.lower()
    ratio = difflib.SequenceMatcher(a=q, b=c).ratio()
    q_toks = tokenize_for_similarity(q)
    c_toks = tokenize_for_similarity(c)
    jacc = 0.0
    if q_toks or c_toks:
        jacc = len(q_toks.intersection(c_toks)) / max(1, len(q_toks.union(c_toks)))
    temporal_bonus = 0.0
    if ("before" in q and "before" in c) or ("after" in q and "after" in c):
        temporal_bonus += 0.05
    return 0.6 * ratio + 0.4 * jacc + temporal_bonus


def detect_temporal_cue(question: str) -> str:
    q = question.lower()
    if "before" in q and "after" in q:
        return "before-after"
    if "before" in q:
        return "before"
    if "after" in q:
        return "after"
    return "none"


def extract_choice(response_text: str, choices: Dict[str, str]) -> Optional[str]:
    if not response_text:
        return None
    valid_keys = set(choices.keys())
    response = response_text.strip()
    match = FINAL_PATTERN.search(response)
    candidate = match.group(1).strip() if match else response
    candidate_clean = candidate.split()[0].strip(".():").upper()
    if candidate_clean in valid_keys:
        return candidate_clean
    for key in valid_keys:
        if re.search(rf"\b{re.escape(key)}\b", response, re.IGNORECASE):
            return key
    lowered = response.lower()
    for key, text in choices.items():
        if text and text.lower() in lowered:
            return key
    return None


def parse_usage(response) -> Tuple[int, int, int]:
    if isinstance(response, dict):
        usage = response.get("usage", {})
        pt = int(usage.get("prompt_tokens", 0) or 0)
        ct = int(usage.get("completion_tokens", 0) or 0)
        tt = int(usage.get("total_tokens", 0) or 0)
    else:
        usage = getattr(response, "usage", None)
        if usage is None:
            return 0, 0, 0
        pt = int(getattr(usage, "prompt_tokens", 0) or 0)
        ct = int(getattr(usage, "completion_tokens", 0) or 0)
        tt = int(getattr(usage, "total_tokens", 0) or 0)
    if tt == 0:
        tt = pt + ct
    return pt, ct, tt


def resolve_pricing(model_name: str, args: argparse.Namespace) -> Dict[str, Optional[float]]:
    if args.input_cost_per_1m is not None or args.output_cost_per_1m is not None:
        if args.input_cost_per_1m is None or args.output_cost_per_1m is None:
            raise ValueError(
                "Both --input-cost-per-1m and --output-cost-per-1m must be provided together."
            )
        return {
            "input": float(args.input_cost_per_1m),
            "output": float(args.output_cost_per_1m),
            "source": "cli_override",
        }
    for prefix, pricing in MODEL_PRICING_PER_1M_TOKENS.items():
        if model_name.startswith(prefix):
            return {
                "input": float(pricing["input"]),
                "output": float(pricing["output"]),
                "source": f"builtin:{prefix}",
            }
    return {"input": None, "output": None, "source": "unavailable"}


def estimate_cost_usd(
    prompt_tokens: int,
    completion_tokens: int,
    input_cost_per_1m: Optional[float],
    output_cost_per_1m: Optional[float],
) -> Optional[float]:
    if input_cost_per_1m is None or output_cost_per_1m is None:
        return None
    return (
        (prompt_tokens / 1_000_000.0) * input_cost_per_1m
        + (completion_tokens / 1_000_000.0) * output_cost_per_1m
    )


def call_baseline(
    client,
    model: str,
    prompt: str,
    temperature: float,
) -> Tuple[str, int, int, int]:
    if OPENAI_SDK_MODE == "modern":
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=128,
            messages=[
                {"role": "system", "content": BASELINE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        text = (response.choices[0].message.content or "").strip()
    else:
        response = openai.ChatCompletion.create(
            model=model,
            temperature=temperature,
            max_tokens=128,
            messages=[
                {"role": "system", "content": BASELINE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        text = (response["choices"][0]["message"]["content"] or "").strip()
    pt, ct, tt = parse_usage(response)
    return text, pt, ct, tt


def run_autogen_chat(
    model: str,
    prompt: str,
    temperature: float,
    max_rounds: int,
) -> Tuple[str, int, int, int, int]:
    try:
        import autogen
    except ImportError as exc:
        raise RuntimeError(
            "AutoGen is not installed. Install dependencies in autogen_cyber_eval/requirements.txt"
        ) from exc

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    config = {"model": model, "api_key": api_key}
    if base_url:
        config["base_url"] = base_url
    llm_config = {
        "config_list": [config],
        "temperature": temperature,
        "timeout": 120,
        "use_cache": False,
    }

    analyst = autogen.AssistantAgent(
        name="ThreatIntelAnalyst",
        llm_config=llm_config,
        system_message=ANALYST_SYSTEM_PROMPT,
    )
    mapper = autogen.AssistantAgent(
        name="ATTACKMapper",
        llm_config=llm_config,
        system_message=MAPPER_SYSTEM_PROMPT,
    )
    responder = autogen.AssistantAgent(
        name="IncidentResponder",
        llm_config=llm_config,
        system_message=RESPONDER_SYSTEM_PROMPT,
    )
    decider = autogen.AssistantAgent(
        name="FinalDecider",
        llm_config=llm_config,
        system_message=DECIDER_SYSTEM_PROMPT,
    )

    user_proxy = autogen.UserProxyAgent(
        name="UserProxy",
        code_execution_config=False,
        human_input_mode="NEVER",
        max_consecutive_auto_reply=max_rounds,
        is_termination_msg=lambda m: "final answer:" in (m.get("content", "").lower()),
    )

    usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    original_chat_create = None
    original_completion_create = None

    if OPENAI_SDK_MODE == "legacy":
        original_chat_create = openai.ChatCompletion.create
        original_completion_create = getattr(openai.Completion, "create", None)

        def tracked_chat_create(*args, **kwargs):
            response = original_chat_create(*args, **kwargs)
            pt, ct, tt = parse_usage(response)
            usage_totals["prompt_tokens"] += pt
            usage_totals["completion_tokens"] += ct
            usage_totals["total_tokens"] += tt
            return response

        def tracked_completion_create(*args, **kwargs):
            response = original_completion_create(*args, **kwargs)
            pt, ct, tt = parse_usage(response)
            usage_totals["prompt_tokens"] += pt
            usage_totals["completion_tokens"] += ct
            usage_totals["total_tokens"] += tt
            return response

        openai.ChatCompletion.create = tracked_chat_create
        if original_completion_create is not None:
            openai.Completion.create = tracked_completion_create

    try:
        try:
            groupchat = autogen.GroupChat(
                agents=[user_proxy, analyst, mapper, responder, decider],
                messages=[],
                max_round=max_rounds,
                speaker_selection_method="round_robin",
            )
        except TypeError:
            groupchat = autogen.GroupChat(
                agents=[user_proxy, analyst, mapper, responder, decider],
                messages=[],
                max_round=max_rounds,
            )
        manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)
        user_proxy.initiate_chat(manager, message=prompt, silent=True)
    finally:
        if original_chat_create is not None:
            openai.ChatCompletion.create = original_chat_create
        if original_completion_create is not None:
            openai.Completion.create = original_completion_create

    all_messages: List[str] = []
    for msg in groupchat.messages:
        content = msg.get("content")
        if isinstance(content, str):
            all_messages.append(content)

    final_text = "\n".join(all_messages).strip() or "Final Answer: "
    return (
        final_text,
        len(groupchat.messages),
        usage_totals["prompt_tokens"],
        usage_totals["completion_tokens"],
        usage_totals["total_tokens"],
    )


def safe_run_with_retry(run_fn, prompt: str):
    try:
        return run_fn(prompt), None
    except Exception as exc:
        if "parse the JSON body of your request" in str(exc):
            try:
                sanitized_prompt = sanitize_prompt_text(prompt, ascii_only=True)
                return run_fn(sanitized_prompt), None
            except Exception as retry_exc:
                return None, f"{type(retry_exc).__name__}: {retry_exc}"
        return None, f"{type(exc).__name__}: {exc}"


def load_memory_records(store_path: Path, load_existing: bool) -> List[Dict[str, object]]:
    if not load_existing or not store_path.exists():
        return []
    rows: List[Dict[str, object]] = []
    with open(store_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def save_memory_records(store_path: Path, rows: List[Dict[str, object]]) -> None:
    store_path.parent.mkdir(parents=True, exist_ok=True)
    with open(store_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def retrieve_memory(
    query_text: str,
    memory_rows: List[Dict[str, object]],
    success_topk: int,
    failed_topk: int,
    min_similarity: float,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    scored_success: List[Tuple[float, Dict[str, object]]] = []
    scored_failed: List[Tuple[float, Dict[str, object]]] = []

    for row in memory_rows:
        candidate = str(row.get("query_text", ""))
        sim = memory_similarity(query_text, candidate)
        if sim < min_similarity:
            continue
        if bool(row.get("success", False)):
            scored_success.append((sim, row))
        else:
            scored_failed.append((sim, row))

    scored_success.sort(key=lambda x: x[0], reverse=True)
    scored_failed.sort(key=lambda x: x[0], reverse=True)

    return (
        [row for _, row in scored_success[:success_topk]],
        [row for _, row in scored_failed[:failed_topk]],
    )


def build_memory_block(success_rows: List[Dict[str, object]], failed_rows: List[Dict[str, object]]) -> str:
    lines: List[str] = []
    if success_rows:
        lines.append("Successful prior cases:")
        for idx, row in enumerate(success_rows, start=1):
            lines.append(
                f"{idx}. cue={row.get('temporal_cue')} answer={row.get('ground_truth')} "
                f"lesson={row.get('lesson', '')}"
            )
    if failed_rows:
        lines.append("Failed prior pitfalls:")
        for idx, row in enumerate(failed_rows, start=1):
            lines.append(
                f"{idx}. predicted={row.get('predicted_answer')} correct={row.get('ground_truth')} "
                f"lesson={row.get('lesson', '')}"
            )
    return "\n".join(lines).strip()


def derive_lesson(question: str, predicted_answer: Optional[str], ground_truth: str, success: bool) -> str:
    cue = detect_temporal_cue(question)
    if success:
        return f"When cue={cue}, align choice with CTI sequence transitions; answer was {ground_truth}."
    return (
        f"Pitfall with cue={cue}: predicted {predicted_answer or 'None'} but correct is {ground_truth}. "
        "Prioritize explicit ordering in the report."
    )


def main() -> None:
    args = parse_args()
    pricing = resolve_pricing(args.model, args)
    load_dotenv_local(args.dotenv_path)

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            f"OPENAI_API_KEY not found. Set it in {args.dotenv_path} or environment variables."
        )
    if not args.dataset_csv.exists():
        raise FileNotFoundError(f"Dataset CSV not found: {args.dataset_csv}")

    if OPENAI_SDK_MODE == "modern":
        client = OpenAI()
    else:
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if os.getenv("OPENAI_BASE_URL"):
            openai.api_base = os.getenv("OPENAI_BASE_URL")
        client = None

    run_start = time.perf_counter()
    task_name = args.dataset_csv.stem

    modes = {args.mode} if args.mode != "all" else {"baseline", "autogen", "gmemory"}

    metrics = {
        "baseline": {
            "total_answered": 0,
            "correct": 0,
            "errors": 0,
            "runtime_seconds": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
        "autogen": {
            "total_answered": 0,
            "correct": 0,
            "errors": 0,
            "runtime_seconds": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "avg_num_messages_acc": 0.0,
        },
        "gmemory": {
            "total_answered": 0,
            "correct": 0,
            "errors": 0,
            "runtime_seconds": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "avg_num_messages_acc": 0.0,
            "retrieved_success_total": 0,
            "retrieved_failed_total": 0,
        },
    }

    memory_rows = load_memory_records(args.gmemory_store, args.gmemory_load_existing)
    results: List[Dict[str, object]] = []

    with open(args.dataset_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            if idx > args.max_questions:
                break

            choices = build_choices(row, task_name)
            ground_truth = normalize_ground_truth(row.get("Ground Truth", ""), choices)
            base_prompt = build_prompt(row, choices)
            question = (row.get("Question") or "").strip()

            row_result: Dict[str, object] = {
                "question_id": (row.get("Question ID") or "").strip(),
                "task_name": task_name,
                "ground_truth": ground_truth,
            }

            # baseline
            baseline_answer: Optional[str] = None
            if "baseline" in modes:
                t0 = time.perf_counter()
                run_fn = lambda p: call_baseline(
                    client=client,
                    model=args.model,
                    prompt=p,
                    temperature=args.temperature,
                )
                payload, err = safe_run_with_retry(run_fn, base_prompt)
                metrics["baseline"]["runtime_seconds"] += time.perf_counter() - t0

                if err is not None:
                    metrics["baseline"]["errors"] += 1
                    row_result["baseline_status"] = "error"
                    row_result["baseline_error"] = err
                else:
                    text, pt, ct, tt = payload
                    choice = extract_choice(text, choices)
                    baseline_answer = choices.get(choice) if choice else None
                    correct = bool(
                        baseline_answer
                        and baseline_answer.strip().casefold() == ground_truth.strip().casefold()
                    )
                    metrics["baseline"]["total_answered"] += 1
                    metrics["baseline"]["correct"] += int(correct)
                    metrics["baseline"]["prompt_tokens"] += pt
                    metrics["baseline"]["completion_tokens"] += ct
                    metrics["baseline"]["total_tokens"] += tt
                    row_result["baseline_status"] = "correct" if correct else "wrong"
                    row_result["baseline_choice"] = choice
                    row_result["baseline_answer"] = baseline_answer

            # autogen
            autogen_answer: Optional[str] = None
            if "autogen" in modes:
                t0 = time.perf_counter()
                run_fn = lambda p: run_autogen_chat(
                    model=args.model,
                    prompt=p,
                    temperature=args.temperature,
                    max_rounds=args.autogen_rounds,
                )
                payload, err = safe_run_with_retry(run_fn, base_prompt)
                metrics["autogen"]["runtime_seconds"] += time.perf_counter() - t0

                if err is not None:
                    metrics["autogen"]["errors"] += 1
                    row_result["autogen_status"] = "error"
                    row_result["autogen_error"] = err
                else:
                    text, num_msgs, pt, ct, tt = payload
                    choice = extract_choice(text, choices)
                    autogen_answer = choices.get(choice) if choice else None
                    correct = bool(
                        autogen_answer
                        and autogen_answer.strip().casefold() == ground_truth.strip().casefold()
                    )
                    metrics["autogen"]["total_answered"] += 1
                    metrics["autogen"]["correct"] += int(correct)
                    metrics["autogen"]["prompt_tokens"] += pt
                    metrics["autogen"]["completion_tokens"] += ct
                    metrics["autogen"]["total_tokens"] += tt
                    metrics["autogen"]["avg_num_messages_acc"] += num_msgs
                    row_result["autogen_status"] = "correct" if correct else "wrong"
                    row_result["autogen_choice"] = choice
                    row_result["autogen_answer"] = autogen_answer

            # gmemory-enhanced autogen
            gmemory_answer: Optional[str] = None
            if "gmemory" in modes:
                query_text = f"Question: {question}\nContext: {(row.get('Context') or '')}"
                success_rows, failed_rows = retrieve_memory(
                    query_text=query_text,
                    memory_rows=memory_rows,
                    success_topk=args.gmemory_success_topk,
                    failed_topk=args.gmemory_failed_topk,
                    min_similarity=args.gmemory_min_similarity,
                )
                metrics["gmemory"]["retrieved_success_total"] += len(success_rows)
                metrics["gmemory"]["retrieved_failed_total"] += len(failed_rows)

                memory_block = build_memory_block(success_rows, failed_rows)
                if memory_block:
                    gmemory_prompt = (
                        f"{base_prompt}\n\n"
                        "Retrieved memory from prior tasks (use as guidance, not strict rules):\n"
                        f"{memory_block}\n\n"
                        "Prioritize current report evidence if memory conflicts."
                    )
                else:
                    gmemory_prompt = base_prompt

                t0 = time.perf_counter()
                run_fn = lambda p: run_autogen_chat(
                    model=args.model,
                    prompt=p,
                    temperature=args.temperature,
                    max_rounds=args.autogen_rounds,
                )
                payload, err = safe_run_with_retry(run_fn, gmemory_prompt)
                metrics["gmemory"]["runtime_seconds"] += time.perf_counter() - t0

                if err is not None:
                    metrics["gmemory"]["errors"] += 1
                    row_result["gmemory_status"] = "error"
                    row_result["gmemory_error"] = err
                else:
                    text, num_msgs, pt, ct, tt = payload
                    choice = extract_choice(text, choices)
                    gmemory_answer = choices.get(choice) if choice else None
                    correct = bool(
                        gmemory_answer
                        and gmemory_answer.strip().casefold() == ground_truth.strip().casefold()
                    )
                    metrics["gmemory"]["total_answered"] += 1
                    metrics["gmemory"]["correct"] += int(correct)
                    metrics["gmemory"]["prompt_tokens"] += pt
                    metrics["gmemory"]["completion_tokens"] += ct
                    metrics["gmemory"]["total_tokens"] += tt
                    metrics["gmemory"]["avg_num_messages_acc"] += num_msgs
                    row_result["gmemory_status"] = "correct" if correct else "wrong"
                    row_result["gmemory_choice"] = choice
                    row_result["gmemory_answer"] = gmemory_answer

                row_result["gmemory_retrieved_success"] = len(success_rows)
                row_result["gmemory_retrieved_failed"] = len(failed_rows)

            # Update memory after each question (online memory growth)
            if "gmemory" in modes:
                source_mode = "gmemory"
                source_answer = gmemory_answer
                source_status = row_result.get("gmemory_status")
                if source_status == "error":
                    if row_result.get("autogen_status") in ("correct", "wrong"):
                        source_mode = "autogen"
                        source_answer = autogen_answer
                        source_status = row_result.get("autogen_status")
                    elif row_result.get("baseline_status") in ("correct", "wrong"):
                        source_mode = "baseline"
                        source_answer = baseline_answer
                        source_status = row_result.get("baseline_status")

                if source_status in ("correct", "wrong"):
                    success = source_status == "correct"
                    memory_rows.append(
                        {
                            "task_name": task_name,
                            "question_id": row_result["question_id"],
                            "query_text": query_text,
                            "ground_truth": ground_truth,
                            "predicted_answer": source_answer,
                            "success": success,
                            "temporal_cue": detect_temporal_cue(question),
                            "lesson": derive_lesson(
                                question=question,
                                predicted_answer=source_answer,
                                ground_truth=ground_truth,
                                success=success,
                            ),
                            "source_mode": source_mode,
                        }
                    )

            results.append(row_result)

            status_parts = [f"[{idx}/{args.max_questions}] qid={row_result['question_id']}"]
            for mode in ["baseline", "autogen", "gmemory"]:
                if mode in modes:
                    status_parts.append(f"{mode}={row_result.get(f'{mode}_status', 'N/A')}")
            print(" ".join(status_parts))

            if args.sleep_between_questions > 0:
                time.sleep(args.sleep_between_questions)

    # save memory snapshot for reuse
    if "gmemory" in modes:
        save_memory_records(args.gmemory_store, memory_rows)

    elapsed = time.perf_counter() - run_start
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_slug = re.sub(r"[^A-Za-z0-9._-]+", "-", args.model)
    results_path = args.output_dir / f"gmemory_results_{model_slug}_{timestamp}.jsonl"
    summary_path = args.output_dir / f"gmemory_summary_{model_slug}_{timestamp}.json"

    with open(results_path, "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_metrics: Dict[str, Dict[str, object]] = {}
    for mode in ["baseline", "autogen", "gmemory"]:
        if mode not in modes:
            continue
        m = metrics[mode]
        answered = int(m["total_answered"])
        correct = int(m["correct"])
        accuracy = (correct / answered) if answered else None
        est_cost = estimate_cost_usd(
            prompt_tokens=int(m["prompt_tokens"]),
            completion_tokens=int(m["completion_tokens"]),
            input_cost_per_1m=pricing["input"],
            output_cost_per_1m=pricing["output"],
        )

        entry: Dict[str, object] = {
            "total_answered": answered,
            "correct": correct,
            "accuracy": accuracy,
            "errors": int(m["errors"]),
            "runtime_seconds": round(float(m["runtime_seconds"]), 3),
            "prompt_tokens": int(m["prompt_tokens"]),
            "completion_tokens": int(m["completion_tokens"]),
            "total_tokens": int(m["total_tokens"]),
            "estimated_cost_usd": est_cost,
        }
        if mode in ("autogen", "gmemory") and answered:
            entry["avg_num_messages"] = float(m["avg_num_messages_acc"]) / answered
        if mode == "gmemory":
            entry["avg_retrieved_success"] = (
                float(m["retrieved_success_total"]) / answered if answered else None
            )
            entry["avg_retrieved_failed"] = (
                float(m["retrieved_failed_total"]) / answered if answered else None
            )
            entry["memory_store_path"] = str(args.gmemory_store.resolve())
            entry["memory_records"] = len(memory_rows)

        summary_metrics[mode] = entry

    deltas: Dict[str, Optional[float]] = {}
    baseline_acc = summary_metrics.get("baseline", {}).get("accuracy")
    autogen_acc = summary_metrics.get("autogen", {}).get("accuracy")
    gmemory_acc = summary_metrics.get("gmemory", {}).get("accuracy")

    if baseline_acc is not None and autogen_acc is not None:
        deltas["autogen_minus_baseline_points"] = round((autogen_acc - baseline_acc) * 100.0, 3)
    if baseline_acc is not None and gmemory_acc is not None:
        deltas["gmemory_minus_baseline_points"] = round((gmemory_acc - baseline_acc) * 100.0, 3)
    if autogen_acc is not None and gmemory_acc is not None:
        deltas["gmemory_minus_autogen_points"] = round((gmemory_acc - autogen_acc) * 100.0, 3)

    summary = {
        "model": args.model,
        "dataset_csv": str(args.dataset_csv.resolve()),
        "mode": args.mode,
        "max_questions": args.max_questions,
        "autogen_rounds": args.autogen_rounds,
        "gmemory": {
            "success_topk": args.gmemory_success_topk,
            "failed_topk": args.gmemory_failed_topk,
            "min_similarity": args.gmemory_min_similarity,
            "load_existing": args.gmemory_load_existing,
            "store_path": str(args.gmemory_store.resolve()),
        },
        "runtime_seconds": round(elapsed, 3),
        "pricing": {
            "source": pricing["source"],
            "input_cost_per_1m": pricing["input"],
            "output_cost_per_1m": pricing["output"],
        },
        "metrics": summary_metrics,
        "deltas": deltas,
        "artifacts": {
            "results_jsonl": str(results_path.resolve()),
            "summary_json": str(summary_path.resolve()),
        },
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== GMemory Cybersecurity Benchmark Summary ===")
    print(f"Task: {task_name}")
    print(f"Runtime (s): {elapsed:.2f}")
    for mode in ["baseline", "autogen", "gmemory"]:
        if mode not in summary_metrics:
            continue
        m = summary_metrics[mode]
        acc_text = "N/A" if m["accuracy"] is None else f"{m['accuracy']:.2%}"
        cost_text = "N/A" if m["estimated_cost_usd"] is None else f"${m['estimated_cost_usd']:.6f}"
        print(
            f"{mode:8} acc={acc_text} answered={m['total_answered']} "
            f"errors={m['errors']} runtime={m['runtime_seconds']:.2f}s "
            f"tokens={m['total_tokens']} est_cost={cost_text}"
        )
    for key, value in deltas.items():
        print(f"{key}: {value:.3f} points")
    print(f"Results saved to: {results_path}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
