#!/usr/bin/env python3
"""Compare baseline single-agent vs AutoGen multi-agent on cybersecurity QA.

Default dataset is AttackSeqBench AttackSeq-Tactic so you can test whether
multi-agent collaboration improves answer quality for your chosen model.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from dataclasses import asdict, dataclass
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
Analyze the attack flow and key indicators from the report.
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


@dataclass
class ExampleResult:
    question_id: str
    task_name: str
    ground_truth: str
    baseline_choice: Optional[str]
    baseline_answer: Optional[str]
    baseline_correct: bool
    baseline_latency_sec: float
    baseline_prompt_tokens: int
    baseline_completion_tokens: int
    baseline_total_tokens: int
    baseline_error: Optional[str]
    autogen_choice: Optional[str]
    autogen_answer: Optional[str]
    autogen_correct: bool
    autogen_latency_sec: float
    autogen_num_messages: int
    autogen_prompt_tokens: int
    autogen_completion_tokens: int
    autogen_total_tokens: int
    autogen_error: Optional[str]


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
        description="Compare baseline and AutoGen multi-agent performance on cybersecurity QA."
    )
    parser.add_argument(
        "--dataset-csv",
        type=Path,
        default=workspace_root / "AttackSeqBench" / "dataset" / "AttackSeq-Tactic.csv",
        help="CSV path for benchmark questions.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model name.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["baseline", "autogen", "both"],
        default="both",
        help="Which pipeline(s) to run.",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=10,
        help="Max questions to evaluate.",
    )
    parser.add_argument(
        "--autogen-rounds",
        type=int,
        default=8,
        help="AutoGen max group chat rounds.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Model sampling temperature.",
    )
    parser.add_argument(
        "--sleep-between-questions",
        type=float,
        default=0.0,
        help="Optional sleep in seconds between questions.",
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
    parser.add_argument(
        "--input-cost-per-1m",
        type=float,
        default=None,
        help="Optional override for input token cost in USD per 1M tokens.",
    )
    parser.add_argument(
        "--output-cost-per-1m",
        type=float,
        default=None,
        help="Optional override for output token cost in USD per 1M tokens.",
    )
    return parser.parse_args()


def is_procedure_task(task_name: str) -> bool:
    return "AttackSeq-Procedure" in task_name


def build_choices(row: Dict[str, str], task_name: str) -> Dict[str, str]:
    if is_procedure_task(task_name):
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
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        total_tokens = int(usage.get("total_tokens", 0) or 0)
    else:
        usage = getattr(response, "usage", None)
        if usage is None:
            return 0, 0, 0
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
    if total_tokens == 0:
        total_tokens = prompt_tokens + completion_tokens
    return prompt_tokens, completion_tokens, total_tokens


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
    input_cost = (prompt_tokens / 1_000_000.0) * input_cost_per_1m
    output_cost = (completion_tokens / 1_000_000.0) * output_cost_per_1m
    return input_cost + output_cost


def run_baseline(
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
        content = (response.choices[0].message.content or "").strip()
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
        content = (response["choices"][0]["message"]["content"] or "").strip()
    prompt_tokens, completion_tokens, total_tokens = parse_usage(response)
    return content, prompt_tokens, completion_tokens, total_tokens


def run_autogen(
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

    original_chat_completion_create = None
    original_completion_create = None
    if OPENAI_SDK_MODE == "legacy":
        original_chat_completion_create = openai.ChatCompletion.create
        original_completion_create = getattr(openai.Completion, "create", None)

        def tracked_chat_completion_create(*args, **kwargs):
            response = original_chat_completion_create(*args, **kwargs)
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

        openai.ChatCompletion.create = tracked_chat_completion_create
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
        user_proxy.initiate_chat(manager, message=prompt)
    finally:
        if original_chat_completion_create is not None:
            openai.ChatCompletion.create = original_chat_completion_create
        if original_completion_create is not None:
            openai.Completion.create = original_completion_create

    all_messages: List[str] = []
    for msg in groupchat.messages:
        content = msg.get("content")
        if isinstance(content, str):
            all_messages.append(content)

    final_text = "\n".join(all_messages).strip()
    if not final_text:
        final_text = "Final Answer: "
    return (
        final_text,
        len(groupchat.messages),
        usage_totals["prompt_tokens"],
        usage_totals["completion_tokens"],
        usage_totals["total_tokens"],
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
    results: List[ExampleResult] = []
    baseline_runtime_total = 0.0
    autogen_runtime_total = 0.0
    baseline_prompt_tokens_total = 0
    baseline_completion_tokens_total = 0
    baseline_total_tokens_total = 0
    autogen_prompt_tokens_total = 0
    autogen_completion_tokens_total = 0
    autogen_total_tokens_total = 0
    baseline_error_count = 0
    autogen_error_count = 0

    with open(args.dataset_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            if idx > args.max_questions:
                break

            choices = build_choices(row, task_name)
            ground_truth = normalize_ground_truth(row.get("Ground Truth", ""), choices)
            prompt = build_prompt(row, choices)

            baseline_choice: Optional[str] = None
            baseline_answer: Optional[str] = None
            baseline_correct = False
            baseline_latency = 0.0
            baseline_prompt_tokens = 0
            baseline_completion_tokens = 0
            baseline_total_tokens = 0
            baseline_error: Optional[str] = None

            autogen_choice: Optional[str] = None
            autogen_answer: Optional[str] = None
            autogen_correct = False
            autogen_latency = 0.0
            autogen_messages = 0
            autogen_prompt_tokens = 0
            autogen_completion_tokens = 0
            autogen_total_tokens = 0
            autogen_error: Optional[str] = None

            if args.mode in ("baseline", "both"):
                t0 = time.perf_counter()
                baseline_model_text = ""
                try:
                    baseline_model_text, pt, ct, tt = run_baseline(
                        client=client,
                        model=args.model,
                        prompt=prompt,
                        temperature=args.temperature,
                    )
                except Exception as exc:
                    if "parse the JSON body of your request" in str(exc):
                        try:
                            sanitized_prompt = sanitize_prompt_text(prompt, ascii_only=True)
                            baseline_model_text, pt, ct, tt = run_baseline(
                                client=client,
                                model=args.model,
                                prompt=sanitized_prompt,
                                temperature=args.temperature,
                            )
                        except Exception as retry_exc:
                            baseline_error = f"{type(retry_exc).__name__}: {retry_exc}"
                            pt = ct = tt = 0
                    else:
                        baseline_error = f"{type(exc).__name__}: {exc}"
                        pt = ct = tt = 0
                baseline_latency = time.perf_counter() - t0
                baseline_runtime_total += baseline_latency
                baseline_prompt_tokens = pt
                baseline_completion_tokens = ct
                baseline_total_tokens = tt
                baseline_prompt_tokens_total += pt
                baseline_completion_tokens_total += ct
                baseline_total_tokens_total += tt
                if baseline_error is None:
                    baseline_choice = extract_choice(baseline_model_text, choices)
                    baseline_answer = choices.get(baseline_choice) if baseline_choice else None
                    baseline_correct = bool(
                        baseline_answer
                        and baseline_answer.strip().casefold() == ground_truth.strip().casefold()
                    )
                else:
                    baseline_error_count += 1
                    print(f"[warn] qid={row.get('Question ID')} baseline failed: {baseline_error}")

            if args.mode in ("autogen", "both"):
                t0 = time.perf_counter()
                try:
                    (
                        autogen_text,
                        autogen_messages,
                        autogen_prompt_tokens,
                        autogen_completion_tokens,
                        autogen_total_tokens,
                    ) = run_autogen(
                        model=args.model,
                        prompt=prompt,
                        temperature=args.temperature,
                        max_rounds=args.autogen_rounds,
                    )
                except Exception as exc:
                    if "parse the JSON body of your request" in str(exc):
                        try:
                            sanitized_prompt = sanitize_prompt_text(prompt, ascii_only=True)
                            (
                                autogen_text,
                                autogen_messages,
                                autogen_prompt_tokens,
                                autogen_completion_tokens,
                                autogen_total_tokens,
                            ) = run_autogen(
                                model=args.model,
                                prompt=sanitized_prompt,
                                temperature=args.temperature,
                                max_rounds=args.autogen_rounds,
                            )
                        except Exception as retry_exc:
                            autogen_error = f"{type(retry_exc).__name__}: {retry_exc}"
                    else:
                        autogen_error = f"{type(exc).__name__}: {exc}"
                autogen_latency = time.perf_counter() - t0
                autogen_runtime_total += autogen_latency
                autogen_prompt_tokens_total += autogen_prompt_tokens
                autogen_completion_tokens_total += autogen_completion_tokens
                autogen_total_tokens_total += autogen_total_tokens
                if autogen_error is None:
                    autogen_choice = extract_choice(autogen_text, choices)
                    autogen_answer = choices.get(autogen_choice) if autogen_choice else None
                    autogen_correct = bool(
                        autogen_answer
                        and autogen_answer.strip().casefold() == ground_truth.strip().casefold()
                    )
                else:
                    autogen_error_count += 1
                    print(f"[warn] qid={row.get('Question ID')} autogen failed: {autogen_error}")

            results.append(
                ExampleResult(
                    question_id=(row.get("Question ID") or "").strip(),
                    task_name=task_name,
                    ground_truth=ground_truth,
                    baseline_choice=baseline_choice,
                    baseline_answer=baseline_answer,
                    baseline_correct=baseline_correct,
                    baseline_latency_sec=round(baseline_latency, 4),
                    baseline_prompt_tokens=baseline_prompt_tokens,
                    baseline_completion_tokens=baseline_completion_tokens,
                    baseline_total_tokens=baseline_total_tokens,
                    baseline_error=baseline_error,
                    autogen_choice=autogen_choice,
                    autogen_answer=autogen_answer,
                    autogen_correct=autogen_correct,
                    autogen_latency_sec=round(autogen_latency, 4),
                    autogen_num_messages=autogen_messages,
                    autogen_prompt_tokens=autogen_prompt_tokens,
                    autogen_completion_tokens=autogen_completion_tokens,
                    autogen_total_tokens=autogen_total_tokens,
                    autogen_error=autogen_error,
                )
            )

            baseline_text = "N/A"
            if args.mode in ("baseline", "both"):
                baseline_text = "error" if baseline_error else ("correct" if baseline_correct else "wrong")
            autogen_text = "N/A"
            if args.mode in ("autogen", "both"):
                autogen_text = "error" if autogen_error else ("correct" if autogen_correct else "wrong")

            print(
                f"[{idx}/{args.max_questions}] qid={row.get('Question ID')} "
                f"baseline={baseline_text} autogen={autogen_text}"
            )

            if args.sleep_between_questions > 0:
                time.sleep(args.sleep_between_questions)

    total = len(results)
    baseline_total = sum(1 for r in results if r.baseline_choice is not None)
    baseline_correct = sum(1 for r in results if r.baseline_correct)
    autogen_total = sum(1 for r in results if r.autogen_choice is not None)
    autogen_correct = sum(1 for r in results if r.autogen_correct)

    baseline_accuracy = (baseline_correct / baseline_total) if baseline_total else None
    autogen_accuracy = (autogen_correct / autogen_total) if autogen_total else None
    baseline_estimated_cost = estimate_cost_usd(
        prompt_tokens=baseline_prompt_tokens_total,
        completion_tokens=baseline_completion_tokens_total,
        input_cost_per_1m=pricing["input"],
        output_cost_per_1m=pricing["output"],
    )
    autogen_estimated_cost = estimate_cost_usd(
        prompt_tokens=autogen_prompt_tokens_total,
        completion_tokens=autogen_completion_tokens_total,
        input_cost_per_1m=pricing["input"],
        output_cost_per_1m=pricing["output"],
    )

    elapsed = time.perf_counter() - run_start
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / f"results_{args.model}_{timestamp}.jsonl"
    summary_path = args.output_dir / f"summary_{args.model}_{timestamp}.json"

    with open(results_path, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")

    summary = {
        "model": args.model,
        "dataset_csv": str(args.dataset_csv.resolve()),
        "mode": args.mode,
        "max_questions": args.max_questions,
        "autogen_rounds": args.autogen_rounds,
        "runtime_seconds": round(elapsed, 3),
        "baseline": {
            "total_answered": baseline_total,
            "correct": baseline_correct,
            "accuracy": baseline_accuracy,
            "errors": baseline_error_count,
            "runtime_seconds": round(baseline_runtime_total, 3),
            "prompt_tokens": baseline_prompt_tokens_total,
            "completion_tokens": baseline_completion_tokens_total,
            "total_tokens": baseline_total_tokens_total,
            "estimated_cost_usd": baseline_estimated_cost,
            "avg_latency_sec": (
                sum(r.baseline_latency_sec for r in results) / len(results) if results else None
            ),
        },
        "autogen": {
            "total_answered": autogen_total,
            "correct": autogen_correct,
            "accuracy": autogen_accuracy,
            "errors": autogen_error_count,
            "runtime_seconds": round(autogen_runtime_total, 3),
            "prompt_tokens": autogen_prompt_tokens_total,
            "completion_tokens": autogen_completion_tokens_total,
            "total_tokens": autogen_total_tokens_total,
            "estimated_cost_usd": autogen_estimated_cost,
            "avg_latency_sec": (
                sum(r.autogen_latency_sec for r in results) / len(results) if results else None
            ),
            "avg_num_messages": (
                sum(r.autogen_num_messages for r in results) / len(results) if results else None
            ),
        },
        "pricing": {
            "source": pricing["source"],
            "input_cost_per_1m": pricing["input"],
            "output_cost_per_1m": pricing["output"],
        },
        "delta": {
            "accuracy_points": (
                round((autogen_accuracy - baseline_accuracy) * 100.0, 3)
                if autogen_accuracy is not None and baseline_accuracy is not None
                else None
            )
        },
        "artifacts": {
            "results_jsonl": str(results_path.resolve()),
            "summary_json": str(summary_path.resolve()),
        },
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== AutoGen Cybersecurity Evaluation Summary ===")
    print(f"Task: {task_name}")
    print(f"Runtime (s): {elapsed:.2f}")
    if baseline_accuracy is not None:
        print(f"Baseline Accuracy: {baseline_correct}/{baseline_total} ({baseline_accuracy:.2%})")
        print(
            f"Baseline Runtime: {baseline_runtime_total:.2f}s | "
            f"Tokens: prompt={baseline_prompt_tokens_total}, "
            f"completion={baseline_completion_tokens_total}, total={baseline_total_tokens_total}"
        )
        if baseline_estimated_cost is not None:
            print(f"Baseline Estimated Cost (USD): ${baseline_estimated_cost:.6f}")
    if autogen_accuracy is not None:
        print(f"AutoGen Accuracy: {autogen_correct}/{autogen_total} ({autogen_accuracy:.2%})")
        print(
            f"AutoGen Runtime: {autogen_runtime_total:.2f}s | "
            f"Tokens: prompt={autogen_prompt_tokens_total}, "
            f"completion={autogen_completion_tokens_total}, total={autogen_total_tokens_total}"
        )
        if autogen_total_tokens_total == 0:
            print("AutoGen token usage is 0. This usually indicates cached/no tracked API calls.")
        if autogen_estimated_cost is not None:
            print(f"AutoGen Estimated Cost (USD): ${autogen_estimated_cost:.6f}")
    if summary["delta"]["accuracy_points"] is not None:
        print(f"Delta (AutoGen - Baseline): {summary['delta']['accuracy_points']:.3f} points")
    print(f"Results saved to: {results_path}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
