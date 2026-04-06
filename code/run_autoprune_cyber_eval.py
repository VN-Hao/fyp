#!/usr/bin/env python3
"""Benchmark baseline vs AutoGen vs AutoPrune-style MAS for cybersecurity QA.

AutoPrune-style mode applies two pruning ideas inspired by AgentPrune:
- Spatial pruning: keep only top-k agent messages per round by utility score.
- Temporal pruning: keep only a bounded recent memory window.
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
Return exactly one line in this format:
Final Answer: <CHOICE>
"""

ANALYST_SYSTEM_PROMPT = """You are ThreatIntelAnalyst.
Analyze attack progression and key CTI evidence.
Provide concise reasoning and a tentative answer.
"""

MAPPER_SYSTEM_PROMPT = """You are ATTACKMapper.
Map the report details to ATT&CK tactics/techniques/procedures.
Provide concise reasoning and a tentative answer.
"""

RESPONDER_SYSTEM_PROMPT = """You are IncidentResponder.
Focus on practical attack plausibility and sequence consistency.
Provide concise reasoning and a tentative answer.
"""

DECIDER_SYSTEM_PROMPT = """You are FinalDecider.
Synthesize prior reasoning and choose the best final answer.
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
        description="Benchmark baseline, AutoGen, and AutoPrune-style MAS on cybersecurity QA."
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
        choices=["baseline", "autogen", "autoprune", "all"],
        default="all",
        help="Which system to run.",
    )
    parser.add_argument("--max-questions", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--autogen-rounds", type=int, default=8)
    parser.add_argument("--autoprune-rounds", type=int, default=2)
    parser.add_argument("--autoprune-keep-top-k", type=int, default=2)
    parser.add_argument("--autoprune-max-memory", type=int, default=4)
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


def call_chat(
    client,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int = 256,
) -> Tuple[str, int, int, int]:
    if OPENAI_SDK_MODE == "modern":
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        text = (response.choices[0].message.content or "").strip()
    else:
        response = openai.ChatCompletion.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        text = (response["choices"][0]["message"]["content"] or "").strip()

    pt, ct, tt = parse_usage(response)
    return text, pt, ct, tt


def run_baseline(client, model: str, prompt: str, temperature: float) -> Tuple[str, int, int, int]:
    return call_chat(
        client=client,
        model=model,
        system_prompt=BASELINE_SYSTEM_PROMPT,
        user_prompt=prompt,
        temperature=temperature,
        max_tokens=128,
    )


def run_autogen(
    model: str,
    prompt: str,
    temperature: float,
    max_rounds: int,
) -> Tuple[str, int, int, int, int]:
    try:
        import autogen
    except ImportError as exc:
        raise RuntimeError("AutoGen is not installed. Install pyautogen first.") from exc

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
        name="ThreatIntelAnalyst", llm_config=llm_config, system_message=ANALYST_SYSTEM_PROMPT
    )
    mapper = autogen.AssistantAgent(
        name="ATTACKMapper", llm_config=llm_config, system_message=MAPPER_SYSTEM_PROMPT
    )
    responder = autogen.AssistantAgent(
        name="IncidentResponder", llm_config=llm_config, system_message=RESPONDER_SYSTEM_PROMPT
    )
    decider = autogen.AssistantAgent(
        name="FinalDecider", llm_config=llm_config, system_message=DECIDER_SYSTEM_PROMPT
    )
    user_proxy = autogen.UserProxyAgent(
        name="UserProxy",
        code_execution_config=False,
        human_input_mode="NEVER",
        max_consecutive_auto_reply=max_rounds,
        is_termination_msg=lambda m: "final answer:" in (m.get("content", "").lower()),
    )

    usage_totals = {"prompt": 0, "completion": 0, "total": 0}
    original_chat_create = None
    original_completion_create = None

    if OPENAI_SDK_MODE == "legacy":
        original_chat_create = openai.ChatCompletion.create
        original_completion_create = getattr(openai.Completion, "create", None)

        def tracked_chat_create(*args, **kwargs):
            response = original_chat_create(*args, **kwargs)
            pt, ct, tt = parse_usage(response)
            usage_totals["prompt"] += pt
            usage_totals["completion"] += ct
            usage_totals["total"] += tt
            return response

        def tracked_completion_create(*args, **kwargs):
            response = original_completion_create(*args, **kwargs)
            pt, ct, tt = parse_usage(response)
            usage_totals["prompt"] += pt
            usage_totals["completion"] += ct
            usage_totals["total"] += tt
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
        usage_totals["prompt"],
        usage_totals["completion"],
        usage_totals["total"],
    )


def candidate_score(text: str, prompt: str, prior_texts: List[str]) -> float:
    text_l = text.lower()
    prompt_words = set(re.findall(r"[a-zA-Z]{4,}", prompt.lower()))
    text_words = set(re.findall(r"[a-zA-Z]{4,}", text_l))

    overlap = 0.0
    if prompt_words:
        overlap = len(prompt_words.intersection(text_words)) / len(prompt_words)

    format_bonus = 0.0
    if "tentative answer" in text_l or "final answer" in text_l:
        format_bonus += 0.5

    reasoning_bonus = 0.2 if ("because" in text_l or "therefore" in text_l) else 0.0

    redundancy_penalty = 0.0
    for prev in prior_texts:
        ratio = difflib.SequenceMatcher(a=text_l, b=prev.lower()).ratio()
        if ratio > 0.9:
            redundancy_penalty += 0.5

    return overlap + format_bonus + reasoning_bonus - redundancy_penalty


def run_autoprune(
    client,
    model: str,
    prompt: str,
    temperature: float,
    rounds: int,
    keep_top_k: int,
    max_memory: int,
) -> Tuple[str, int, int, int, int]:
    agents = [
        ("ThreatIntelAnalyst", ANALYST_SYSTEM_PROMPT),
        ("ATTACKMapper", MAPPER_SYSTEM_PROMPT),
        ("IncidentResponder", RESPONDER_SYSTEM_PROMPT),
    ]

    memory: List[Dict[str, object]] = []
    prompt_tokens_total = 0
    completion_tokens_total = 0
    total_tokens_total = 0

    for _ in range(max(1, rounds)):
        candidates: List[Dict[str, object]] = []

        context_lines = []
        for m in memory[-max_memory:]:
            context_lines.append(f"{m['agent']}: {m['text']}")
        context_block = "\n".join(context_lines)

        for agent_name, system_prompt in agents:
            user_prompt = (
                f"{prompt}\n\n"
                f"Pruned team memory:\n{context_block}\n\n"
                "Respond with concise reasoning and one line: "
                "Tentative Answer: <CHOICE>"
            )
            text, pt, ct, tt = call_chat(
                client=client,
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=220,
            )
            prompt_tokens_total += pt
            completion_tokens_total += ct
            total_tokens_total += tt
            candidates.append({"agent": agent_name, "text": text})

        prior_texts = [str(m["text"]) for m in memory]
        for c in candidates:
            c["score"] = candidate_score(str(c["text"]), prompt, prior_texts)

        candidates.sort(key=lambda x: float(x["score"]), reverse=True)
        keep_n = max(1, min(keep_top_k, len(candidates)))
        kept = candidates[:keep_n]

        memory.extend(kept)
        if len(memory) > max_memory:
            memory = memory[-max_memory:]

    pruned_context = "\n".join(
        [f"{m['agent']} (score={float(m['score']):.3f}): {m['text']}" for m in memory]
    )
    final_user_prompt = (
        f"{prompt}\n\n"
        f"Pruned messages:\n{pruned_context}\n\n"
        "Return exactly one line: Final Answer: <CHOICE>"
    )
    final_text, pt, ct, tt = call_chat(
        client=client,
        model=model,
        system_prompt=DECIDER_SYSTEM_PROMPT,
        user_prompt=final_user_prompt,
        temperature=temperature,
        max_tokens=128,
    )
    prompt_tokens_total += pt
    completion_tokens_total += ct
    total_tokens_total += tt

    return final_text, len(memory), prompt_tokens_total, completion_tokens_total, total_tokens_total


def safe_run_with_retry(func, prompt: str):
    try:
        return func(prompt), None
    except Exception as exc:
        if "parse the JSON body of your request" in str(exc):
            try:
                sanitized_prompt = sanitize_prompt_text(prompt, ascii_only=True)
                return func(sanitized_prompt), None
            except Exception as retry_exc:
                return None, f"{type(retry_exc).__name__}: {retry_exc}"
        return None, f"{type(exc).__name__}: {exc}"


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

    modes = {args.mode} if args.mode != "all" else {"baseline", "autogen", "autoprune"}
    run_start = time.perf_counter()

    stats = {
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
        "autoprune": {
            "total_answered": 0,
            "correct": 0,
            "errors": 0,
            "runtime_seconds": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "avg_num_messages_acc": 0.0,
        },
    }

    task_name = args.dataset_csv.stem
    results: List[Dict[str, object]] = []

    with open(args.dataset_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            if idx > args.max_questions:
                break

            choices = build_choices(row, task_name)
            gt = normalize_ground_truth(row.get("Ground Truth", ""), choices)
            prompt = build_prompt(row, choices)

            row_out: Dict[str, object] = {
                "question_id": (row.get("Question ID") or "").strip(),
                "task_name": task_name,
                "ground_truth": gt,
            }

            for mode in ["baseline", "autogen", "autoprune"]:
                if mode not in modes:
                    continue

                t0 = time.perf_counter()

                if mode == "baseline":
                    run_fn = lambda p: run_baseline(
                        client=client, model=args.model, prompt=p, temperature=args.temperature
                    )
                    payload, err = safe_run_with_retry(run_fn, prompt)
                    elapsed = time.perf_counter() - t0
                    stats[mode]["runtime_seconds"] += elapsed
                    if err is not None:
                        stats[mode]["errors"] += 1
                        row_out[f"{mode}_status"] = "error"
                        row_out[f"{mode}_error"] = err
                        continue

                    text, pt, ct, tt = payload
                    stats[mode]["prompt_tokens"] += pt
                    stats[mode]["completion_tokens"] += ct
                    stats[mode]["total_tokens"] += tt
                    choice = extract_choice(text, choices)

                elif mode == "autogen":
                    run_fn = lambda p: run_autogen(
                        model=args.model,
                        prompt=p,
                        temperature=args.temperature,
                        max_rounds=args.autogen_rounds,
                    )
                    payload, err = safe_run_with_retry(run_fn, prompt)
                    elapsed = time.perf_counter() - t0
                    stats[mode]["runtime_seconds"] += elapsed
                    if err is not None:
                        stats[mode]["errors"] += 1
                        row_out[f"{mode}_status"] = "error"
                        row_out[f"{mode}_error"] = err
                        continue

                    text, num_msgs, pt, ct, tt = payload
                    stats[mode]["avg_num_messages_acc"] += num_msgs
                    stats[mode]["prompt_tokens"] += pt
                    stats[mode]["completion_tokens"] += ct
                    stats[mode]["total_tokens"] += tt
                    choice = extract_choice(text, choices)

                else:
                    run_fn = lambda p: run_autoprune(
                        client=client,
                        model=args.model,
                        prompt=p,
                        temperature=args.temperature,
                        rounds=args.autoprune_rounds,
                        keep_top_k=args.autoprune_keep_top_k,
                        max_memory=args.autoprune_max_memory,
                    )
                    payload, err = safe_run_with_retry(run_fn, prompt)
                    elapsed = time.perf_counter() - t0
                    stats[mode]["runtime_seconds"] += elapsed
                    if err is not None:
                        stats[mode]["errors"] += 1
                        row_out[f"{mode}_status"] = "error"
                        row_out[f"{mode}_error"] = err
                        continue

                    text, num_msgs, pt, ct, tt = payload
                    stats[mode]["avg_num_messages_acc"] += num_msgs
                    stats[mode]["prompt_tokens"] += pt
                    stats[mode]["completion_tokens"] += ct
                    stats[mode]["total_tokens"] += tt
                    choice = extract_choice(text, choices)

                answer = choices.get(choice) if choice else None
                is_correct = bool(answer and answer.strip().casefold() == gt.strip().casefold())

                stats[mode]["total_answered"] += 1
                stats[mode]["correct"] += int(is_correct)
                row_out[f"{mode}_status"] = "correct" if is_correct else "wrong"
                row_out[f"{mode}_choice"] = choice
                row_out[f"{mode}_answer"] = answer

            results.append(row_out)

            parts = [f"[{idx}/{args.max_questions}] qid={row_out['question_id']}"]
            for mode in ["baseline", "autogen", "autoprune"]:
                if mode in modes:
                    parts.append(f"{mode}={row_out.get(f'{mode}_status', 'N/A')}")
            print(" ".join(parts))

            if args.sleep_between_questions > 0:
                time.sleep(args.sleep_between_questions)

    elapsed_total = time.perf_counter() - run_start
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_slug = re.sub(r"[^A-Za-z0-9._-]+", "-", args.model)
    results_path = args.output_dir / f"autoprune_results_{model_slug}_{timestamp}.jsonl"
    summary_path = args.output_dir / f"autoprune_summary_{model_slug}_{timestamp}.json"

    with open(results_path, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    summary_modes: Dict[str, object] = {}
    for mode in ["baseline", "autogen", "autoprune"]:
        if mode not in modes:
            continue
        answered = int(stats[mode]["total_answered"])
        correct = int(stats[mode]["correct"])
        acc = (correct / answered) if answered else None
        est_cost = estimate_cost_usd(
            prompt_tokens=int(stats[mode]["prompt_tokens"]),
            completion_tokens=int(stats[mode]["completion_tokens"]),
            input_cost_per_1m=pricing["input"],
            output_cost_per_1m=pricing["output"],
        )

        entry: Dict[str, object] = {
            "total_answered": answered,
            "correct": correct,
            "accuracy": acc,
            "errors": int(stats[mode]["errors"]),
            "runtime_seconds": round(float(stats[mode]["runtime_seconds"]), 3),
            "prompt_tokens": int(stats[mode]["prompt_tokens"]),
            "completion_tokens": int(stats[mode]["completion_tokens"]),
            "total_tokens": int(stats[mode]["total_tokens"]),
            "estimated_cost_usd": est_cost,
        }
        if mode in ("autogen", "autoprune") and answered:
            entry["avg_num_messages"] = float(stats[mode]["avg_num_messages_acc"]) / answered
        summary_modes[mode] = entry

    deltas: Dict[str, Optional[float]] = {}
    baseline_acc = None
    if "baseline" in summary_modes:
        baseline_acc = summary_modes["baseline"].get("accuracy")
    for mode in ["autogen", "autoprune"]:
        if mode in summary_modes and baseline_acc is not None and summary_modes[mode].get("accuracy") is not None:
            deltas[f"{mode}_minus_baseline_points"] = round(
                (summary_modes[mode]["accuracy"] - baseline_acc) * 100.0, 3
            )

    summary = {
        "model": args.model,
        "dataset_csv": str(args.dataset_csv.resolve()),
        "mode": args.mode,
        "max_questions": args.max_questions,
        "autogen_rounds": args.autogen_rounds,
        "autoprune": {
            "rounds": args.autoprune_rounds,
            "keep_top_k": args.autoprune_keep_top_k,
            "max_memory": args.autoprune_max_memory,
        },
        "runtime_seconds": round(elapsed_total, 3),
        "pricing": {
            "source": pricing["source"],
            "input_cost_per_1m": pricing["input"],
            "output_cost_per_1m": pricing["output"],
        },
        "metrics": summary_modes,
        "deltas": deltas,
        "artifacts": {
            "results_jsonl": str(results_path.resolve()),
            "summary_json": str(summary_path.resolve()),
        },
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== AutoPrune Cybersecurity Benchmark Summary ===")
    print(f"Task: {Path(args.dataset_csv).stem}")
    print(f"Total Runtime: {elapsed_total:.2f}s")
    for mode in ["baseline", "autogen", "autoprune"]:
        if mode not in summary_modes:
            continue
        m = summary_modes[mode]
        acc_text = "N/A" if m["accuracy"] is None else f"{m['accuracy']:.2%}"
        cost_text = "N/A" if m["estimated_cost_usd"] is None else f"${m['estimated_cost_usd']:.6f}"
        print(
            f"{mode:9} acc={acc_text} answered={m['total_answered']} "
            f"errors={m['errors']} runtime={m['runtime_seconds']:.2f}s "
            f"tokens={m['total_tokens']} est_cost={cost_text}"
        )
    for k, v in deltas.items():
        print(f"{k}: {v:.3f} points")
    print(f"Results saved to: {results_path}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
