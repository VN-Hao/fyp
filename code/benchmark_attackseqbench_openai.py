#!/usr/bin/env python3
"""Benchmark an OpenAI model on AttackSeqBench.

This script is intentionally standalone and placed outside the AttackSeqBench
folder so it can be run from the workspace root.
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
from typing import Any, Dict, List, Optional, Tuple

try:
    from openai import OpenAI  # OpenAI SDK v1+

    OPENAI_SDK_MODE = "modern"
except ImportError:
    import openai  # OpenAI SDK v0.x

    OpenAI = None
    OPENAI_SDK_MODE = "legacy"


SYSTEM_PROMPT = """You are a cybersecurity expert with deep knowledge of Cyber Threat Intelligence (CTI) reports and MITRE ATT&CK.

You will be given:
1) A CTI report context.
2) A question and answer choices.

Pick the single best answer choice based on the report.

Output format requirement:
- Return exactly one line in this format: Final Answer: <LETTER>
- <LETTER> must be one of the provided choice letters (for example A, B, C, D).
"""

CASCADE_DIAGNOSTIC_SUFFIX = """

Return a strict JSON object with this schema and nothing else:
{
    "final_answer": "A|B|C|D",
    "confidence": 0-100,
    "scores": {"A": 0-100, "B": 0-100, "C": 0-100, "D": 0-100},
    "brief_reason": "one short sentence"
}
"""

VERIFIER_SYSTEM_PROMPT = """You are a second-pass cybersecurity verifier.

Independently choose the best answer from the provided options.
Return exactly one line in this format: Final Answer: <LETTER>
"""

FINAL_ANSWER_PATTERN = re.compile(r"final\s*answer\s*:\s*([A-D])", re.IGNORECASE)
LETTER_PATTERN = re.compile(r"\b([A-D])\b")

# Estimated USD pricing per 1M tokens.
# Override via CLI arguments if you want exact billing assumptions.
MODEL_PRICING_PER_1M_TOKENS = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
}


def _tokenize_text(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]{3,}", text.lower()))


class MitreKnowledgeBaseRetriever:
    def __init__(self, kb_path: Path, top_k: int = 6, max_chars: int = 1800) -> None:
        self.kb_path = kb_path
        self.top_k = max(1, int(top_k))
        self.max_chars = max(200, int(max_chars))
        self.entries: List[Dict[str, Any]] = []
        self.tactic_by_name: Dict[str, Dict[str, Any]] = {}
        self.technique_by_name: Dict[str, List[Dict[str, Any]]] = {}
        self._load()

    def _normalize_examples(self, examples: Any, max_items: int = 2) -> str:
        if not isinstance(examples, list) or not examples:
            return ""
        normalized: List[str] = []
        for item in examples[:max_items]:
            if isinstance(item, str):
                normalized.append(item.strip())
            elif isinstance(item, dict):
                text = item.get("description") or item.get("text") or item.get("name")
                if text:
                    normalized.append(str(text).strip())
        return " ".join(part for part in normalized if part)

    def _load(self) -> None:
        with open(self.kb_path, "r", encoding="utf-8") as f:
            kb = json.load(f)

        for tactic in kb.get("tactics", []):
            tactic_name = str(tactic.get("name", "")).strip()
            tactic_desc = str(tactic.get("description", "")).strip()
            tactic_id = str(tactic.get("id", "")).strip()
            if not tactic_name:
                continue

            tactic_text = (
                f"Tactic {tactic_name} ({tactic_id}): {tactic_desc}".strip()
            )
            tactic_entry = {
                "kind": "tactic",
                "name": tactic_name,
                "id": tactic_id,
                "text": tactic_text,
                "tokens": _tokenize_text(tactic_text),
            }
            self.entries.append(tactic_entry)
            self.tactic_by_name[tactic_name.casefold()] = tactic_entry

            for tech in tactic.get("techniques", []):
                tech_name = str(tech.get("name", "")).strip()
                tech_id = str(tech.get("id", "")).strip()
                tech_desc = str(tech.get("description", "")).strip()
                tech_detail = str(tech.get("detailed_description", "")).strip()
                tech_examples = self._normalize_examples(tech.get("examples"))
                if not tech_name:
                    continue
                parts = [
                    f"Technique {tech_name} ({tech_id}) under {tactic_name}.",
                    tech_desc,
                    tech_detail,
                    f"Examples: {tech_examples}" if tech_examples else "",
                ]
                tech_text = " ".join(p for p in parts if p).strip()
                tech_entry = {
                    "kind": "technique",
                    "name": tech_name,
                    "id": tech_id,
                    "tactic": tactic_name,
                    "text": tech_text,
                    "tokens": _tokenize_text(tech_text),
                }
                self.entries.append(tech_entry)
                self.technique_by_name.setdefault(tech_name.casefold(), []).append(tech_entry)

    def _score_entry(self, entry: Dict[str, Any], query_tokens: set[str], query_text_cf: str) -> float:
        entry_tokens = entry.get("tokens", set())
        jacc = 0.0
        if query_tokens or entry_tokens:
            jacc = len(query_tokens.intersection(entry_tokens)) / max(1, len(query_tokens.union(entry_tokens)))

        lexical_bonus = 0.0
        name_cf = str(entry.get("name", "")).casefold()
        if name_cf and name_cf in query_text_cf:
            lexical_bonus += 0.3
        if str(entry.get("kind", "")) == "tactic" and "tactic" in query_text_cf:
            lexical_bonus += 0.05
        if str(entry.get("kind", "")) == "technique" and "technique" in query_text_cf:
            lexical_bonus += 0.05
        return jacc + lexical_bonus

    def build_context_block(
        self,
        task_name: str,
        row: Dict[str, str],
        choices: Dict[str, str],
    ) -> str:
        selected: List[Dict[str, Any]] = []
        seen: set[Tuple[str, str]] = set()

        def add_entry(entry: Dict[str, Any]) -> None:
            key = (str(entry.get("kind", "")), str(entry.get("id", "")))
            if key not in seen:
                seen.add(key)
                selected.append(entry)

        if "AttackSeq-Tactic" in task_name:
            for _, text in choices.items():
                match = self.tactic_by_name.get((text or "").casefold())
                if match:
                    add_entry(match)
        elif "AttackSeq-Technique" in task_name:
            for _, text in choices.items():
                for match in self.technique_by_name.get((text or "").casefold(), []):
                    add_entry(match)

        context = str(row.get("Context", "")).strip()
        question = str(row.get("Question", "")).strip()
        key_points = str(row.get("Key Points", "")).strip()
        query_text = "\n".join([context, question, key_points, " ".join(choices.values())])
        query_cf = query_text.casefold()
        query_tokens = _tokenize_text(query_text)

        scored: List[Tuple[float, Dict[str, Any]]] = []
        for entry in self.entries:
            key = (str(entry.get("kind", "")), str(entry.get("id", "")))
            if key in seen:
                continue
            score = self._score_entry(entry, query_tokens, query_cf)
            scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        for score, entry in scored[: self.top_k]:
            if score <= 0:
                continue
            add_entry(entry)

        if not selected:
            return ""

        lines = ["Relevant MITRE ATT&CK knowledge (retrieved):"]
        for idx, entry in enumerate(selected[: self.top_k], start=1):
            text = str(entry.get("text", "")).replace("\n", " ").strip()
            if len(text) > 300:
                text = text[:297].rstrip() + "..."
            lines.append(f"{idx}. {text}")

        block = "\n".join(lines).strip()
        if len(block) > self.max_chars:
            block = block[: self.max_chars].rstrip() + "..."
        return block


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


@dataclass
class Prediction:
    task: str
    question_id: str
    attackseq_id: str
    ground_truth: str
    predicted_choice: Optional[str]
    predicted_answer: Optional[str]
    is_correct: bool
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    response_text: str
    model_used: str
    escalated: bool = False
    escalation_reason: Optional[str] = None
    confusion_score: Optional[float] = None
    primary_choice: Optional[str] = None
    primary_confidence: Optional[float] = None
    primary_margin: Optional[float] = None
    vote_share: Optional[float] = None
    verifier_choice: Optional[str] = None


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Benchmark OpenAI chat models on AttackSeqBench CSV files."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=root_dir / "AttackSeqBench" / "dataset",
        help="Directory containing AttackSeqBench CSV files.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Model name, for example gpt-4o-mini.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root_dir / "benchmark_results",
        help="Directory where predictions and summary will be saved.",
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help=(
            "Optional task filter by CSV stem, e.g. "
            "AttackSeq-Tactic AttackSeq-Technique"
        ),
    )
    parser.add_argument(
        "--max-questions-per-task",
        type=int,
        default=None,
        help="Optional cap to run only first N questions from each task.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Max retries per request when transient API errors occur.",
    )
    parser.add_argument(
        "--request-max-tokens",
        type=int,
        default=128,
        help="Max completion tokens for each model response.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (ignored for reasoning-only models).",
    )
    parser.add_argument(
        "--sleep-between-requests",
        type=float,
        default=0.0,
        help="Optional fixed sleep in seconds after each request.",
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
    parser.add_argument(
        "--enable-mitre-kb",
        action="store_true",
        help="Inject retrieved MITRE ATT&CK snippets into each question prompt.",
    )
    parser.add_argument(
        "--mitre-kb-path",
        type=Path,
        default=root_dir / "AttackSeqBench" / "mitre_kb" / "mitre.json",
        help="Path to MITRE ATT&CK knowledge base JSON.",
    )
    parser.add_argument(
        "--mitre-top-k",
        type=int,
        default=6,
        help="Number of MITRE snippets to inject per question.",
    )
    parser.add_argument(
        "--mitre-max-chars",
        type=int,
        default=1800,
        help="Maximum characters for injected MITRE context block.",
    )
    parser.add_argument(
        "--enable-cascade",
        action="store_true",
        help="Enable two-stage cascade: primary model first, escalate uncertain cases.",
    )
    parser.add_argument(
        "--cascade-model",
        type=str,
        default="gpt-4o",
        help="Escalation model for uncertain/high-confusion cases.",
    )
    parser.add_argument(
        "--cascade-max-escalation-rate",
        type=float,
        default=0.25,
        help="Maximum fraction of questions that may escalate to cascade model.",
    )
    parser.add_argument(
        "--cascade-min-confidence",
        type=float,
        default=70.0,
        help="Escalate if primary confidence is below this threshold.",
    )
    parser.add_argument(
        "--cascade-min-margin",
        type=float,
        default=8.0,
        help="Escalate if score margin (top1-top2) is below this threshold.",
    )
    parser.add_argument(
        "--cascade-hard-confidence",
        type=float,
        default=55.0,
        help="Hard escalate if primary confidence is below this threshold.",
    )
    parser.add_argument(
        "--cascade-hard-margin",
        type=float,
        default=5.0,
        help="Hard escalate if score margin is below this threshold.",
    )
    parser.add_argument(
        "--cascade-self-consistency-votes",
        type=int,
        default=3,
        help="Number of primary-model votes for self-consistency.",
    )
    parser.add_argument(
        "--cascade-self-consistency-temperature",
        type=float,
        default=0.2,
        help="Temperature for additional self-consistency votes.",
    )
    parser.add_argument(
        "--cascade-min-vote-share",
        type=float,
        default=0.67,
        help="Escalate if majority vote share is below this threshold.",
    )
    parser.add_argument(
        "--cascade-confusion-threshold",
        type=float,
        default=0.45,
        help="Escalate if weighted confusion score exceeds this threshold.",
    )
    parser.add_argument(
        "--cascade-enable-verifier",
        action="store_true",
        help="Enable one extra verifier pass on primary model before escalation decision.",
    )
    parser.add_argument(
        "--cascade-diagnostic-max-tokens",
        type=int,
        default=220,
        help="Max completion tokens for the primary diagnostic JSON pass.",
    )
    return parser.parse_args()


def is_reasoning_model(model_name: str) -> bool:
    return model_name.startswith(("o1", "o3", "o4"))


def build_answer_choices(row: Dict[str, str], task_name: str) -> Dict[str, str]:
    if "AttackSeq-Procedure" in task_name:
        return {"A": "Yes", "B": "No"}
    return {
        "A": row.get("A", "").strip(),
        "B": row.get("B", "").strip(),
        "C": row.get("C", "").strip(),
        "D": row.get("D", "").strip(),
    }


def normalize_ground_truth(ground_truth: str, choices: Dict[str, str]) -> str:
    gt = (ground_truth or "").strip()
    if len(gt) == 1 and gt.upper() in choices:
        return choices[gt.upper()]
    return gt


def build_user_prompt(
    row: Dict[str, str],
    choices: Dict[str, str],
    mitre_context_block: Optional[str] = None,
) -> str:
    context = (row.get("Context") or "").strip()
    question = (row.get("Question") or "").strip()
    choice_lines = "\n".join(f"{letter}: {text}" for letter, text in choices.items())
    prompt = (
        f"Report: {context}\n"
        f"Question: {question}\n"
        f"Answer Choices:\n{choice_lines}\n\n"
        "Return exactly one line: Final Answer: <LETTER>"
    )
    if mitre_context_block:
        prompt = (
            f"{prompt}\n\n"
            f"{mitre_context_block}\n\n"
            "Use the report as primary evidence and MITRE knowledge as supporting context."
        )
    return prompt


def get_usage_counters(response) -> Dict[str, int]:
    if isinstance(response, dict):
        usage = response.get("usage", {})
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        total_tokens = int(usage.get("total_tokens", 0) or 0)
    else:
        usage = getattr(response, "usage", None)
        if usage is None:
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
    if total_tokens == 0:
        total_tokens = prompt_tokens + completion_tokens
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


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


def estimate_mixed_cost_usd(
    primary_prompt_tokens: int,
    primary_completion_tokens: int,
    primary_pricing: Dict[str, Optional[float]],
    cascade_prompt_tokens: int,
    cascade_completion_tokens: int,
    cascade_pricing: Dict[str, Optional[float]],
) -> Optional[float]:
    primary_cost = estimate_cost_usd(
        prompt_tokens=primary_prompt_tokens,
        completion_tokens=primary_completion_tokens,
        input_cost_per_1m=primary_pricing["input"],
        output_cost_per_1m=primary_pricing["output"],
    )
    cascade_cost = estimate_cost_usd(
        prompt_tokens=cascade_prompt_tokens,
        completion_tokens=cascade_completion_tokens,
        input_cost_per_1m=cascade_pricing["input"],
        output_cost_per_1m=cascade_pricing["output"],
    )
    if primary_cost is None or cascade_cost is None:
        return None
    return primary_cost + cascade_cost


def format_duration(seconds: float) -> str:
    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def call_model(
    client,
    model_name: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    max_retries: int,
    system_prompt: str = SYSTEM_PROMPT,
) -> tuple[str, Dict[str, int]]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    payload = {"model": model_name, "messages": messages}
    if OPENAI_SDK_MODE == "modern" and is_reasoning_model(model_name):
        payload["max_completion_tokens"] = max_tokens
    else:
        payload["max_tokens"] = max_tokens
        payload["temperature"] = temperature

    for attempt in range(1, max_retries + 1):
        try:
            if OPENAI_SDK_MODE == "modern":
                response = client.chat.completions.create(**payload)
                response_text = (response.choices[0].message.content or "").strip()
            else:
                response = openai.ChatCompletion.create(**payload)
                response_text = (response["choices"][0]["message"]["content"] or "").strip()
            usage_counters = get_usage_counters(response)
            return response_text, usage_counters
        except Exception as exc:
            if attempt == max_retries:
                raise RuntimeError(
                    f"Request failed after {max_retries} attempts: {exc}"
                ) from exc
            delay_seconds = min(30, 2 ** (attempt - 1))
            print(
                f"Request failed on attempt {attempt}/{max_retries}. "
                f"Retrying in {delay_seconds}s..."
            )
            time.sleep(delay_seconds)
    return "", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def extract_choice(response_text: str, choices: Dict[str, str]) -> Optional[str]:
    if not response_text:
        return None

    final_match = FINAL_ANSWER_PATTERN.search(response_text)
    if final_match:
        letter = final_match.group(1).upper()
        if letter in choices:
            return letter

    for token in LETTER_PATTERN.findall(response_text.upper()):
        letter = token.upper()
        if letter in choices:
            return letter

    lowered = response_text.lower()
    for letter, answer_text in choices.items():
        if answer_text and answer_text.lower() in lowered:
            return letter

    return None


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    payload = text.strip()
    try:
        obj = json.loads(payload)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    start = payload.find("{")
    end = payload.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        obj = json.loads(payload[start : end + 1])
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


def _clamp_0_100(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return max(0.0, min(100.0, float(value)))


def parse_diagnostic_payload(
    response_text: str,
    choices: Dict[str, str],
) -> Dict[str, Any]:
    obj = _extract_json_object(response_text)
    if obj is None:
        return {
            "parse_ok": False,
            "choice": None,
            "confidence": None,
            "margin": None,
            "scores": {},
        }

    choice_raw = str(obj.get("final_answer", "")).strip().upper()
    choice = choice_raw if choice_raw in choices else None

    confidence = None
    conf_raw = obj.get("confidence")
    if isinstance(conf_raw, (int, float, str)):
        try:
            confidence = _clamp_0_100(float(conf_raw))
        except Exception:
            confidence = None

    score_map: Dict[str, float] = {}
    raw_scores = obj.get("scores")
    if isinstance(raw_scores, dict):
        for letter in choices.keys():
            val = raw_scores.get(letter)
            if isinstance(val, (int, float, str)):
                try:
                    score_map[letter] = _clamp_0_100(float(val)) or 0.0
                except Exception:
                    continue

    margin = None
    if len(score_map) >= 2:
        ordered = sorted(score_map.values(), reverse=True)
        margin = float(ordered[0] - ordered[1])

    return {
        "parse_ok": choice is not None,
        "choice": choice,
        "confidence": confidence,
        "margin": margin,
        "scores": score_map,
    }


def majority_vote(choices: List[Optional[str]]) -> Tuple[Optional[str], Optional[float]]:
    counts: Dict[str, int] = {}
    total = 0
    for choice in choices:
        if not choice:
            continue
        counts[choice] = counts.get(choice, 0) + 1
        total += 1

    if total == 0 or not counts:
        return None, None

    winner = max(counts.items(), key=lambda x: x[1])[0]
    vote_share = counts[winner] / total
    return winner, vote_share


def run_cascade_inference(
    client,
    prompt: str,
    choices: Dict[str, str],
    primary_model: str,
    cascade_model: str,
    max_tokens: int,
    max_retries: int,
    temperature: float,
    question_index_1based: int,
    escalations_so_far: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    diagnostic_prompt = prompt + CASCADE_DIAGNOSTIC_SUFFIX

    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    primary_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    cascade_usage_only = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    diagnostic_text, diagnostic_usage = call_model(
        client=client,
        model_name=primary_model,
        prompt=diagnostic_prompt,
        max_tokens=max(args.cascade_diagnostic_max_tokens, max_tokens),
        temperature=0.0,
        max_retries=max_retries,
    )
    for key in total_usage:
        total_usage[key] += diagnostic_usage[key]
        primary_usage[key] += diagnostic_usage[key]

    diagnostic = parse_diagnostic_payload(diagnostic_text, choices)
    primary_choice = diagnostic["choice"]
    if primary_choice is None:
        primary_choice = extract_choice(diagnostic_text, choices)
        diagnostic["parse_ok"] = bool(primary_choice)

    primary_confidence = diagnostic["confidence"]
    primary_margin = diagnostic["margin"]

    votes: List[Optional[str]] = [primary_choice]
    vote_count = max(1, int(args.cascade_self_consistency_votes))
    for _ in range(vote_count - 1):
        vote_text, vote_usage = call_model(
            client=client,
            model_name=primary_model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=args.cascade_self_consistency_temperature,
            max_retries=max_retries,
        )
        for key in total_usage:
            total_usage[key] += vote_usage[key]
            primary_usage[key] += vote_usage[key]
        votes.append(extract_choice(vote_text, choices))

    majority_choice, vote_share = majority_vote(votes)

    verifier_choice = None
    if args.cascade_enable_verifier:
        verifier_text, verifier_usage = call_model(
            client=client,
            model_name=primary_model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.0,
            max_retries=max_retries,
            system_prompt=VERIFIER_SYSTEM_PROMPT,
        )
        for key in total_usage:
            total_usage[key] += verifier_usage[key]
            primary_usage[key] += verifier_usage[key]
        verifier_choice = extract_choice(verifier_text, choices)

    fallback_choice = majority_choice or primary_choice

    confusion_reasons: List[str] = []
    if not diagnostic["parse_ok"]:
        confusion_reasons.append("format_uncertainty")
    if primary_confidence is not None and primary_confidence < args.cascade_min_confidence:
        confusion_reasons.append("low_confidence")
    if primary_margin is not None and primary_margin < args.cascade_min_margin:
        confusion_reasons.append("low_margin")
    if vote_share is not None and vote_share < args.cascade_min_vote_share:
        confusion_reasons.append("low_consistency")
    if args.cascade_enable_verifier and verifier_choice and fallback_choice and verifier_choice != fallback_choice:
        confusion_reasons.append("verifier_disagreement")
    if fallback_choice is None:
        confusion_reasons.append("no_choice")

    conf_component = 1.0 if primary_confidence is None else (100.0 - primary_confidence) / 100.0
    margin_component = 1.0 if primary_margin is None else (100.0 - max(0.0, min(100.0, primary_margin))) / 100.0
    vote_component = 1.0 if vote_share is None else (1.0 - vote_share)
    format_component = 1.0 if "format_uncertainty" in confusion_reasons else 0.0
    verifier_component = 1.0 if "verifier_disagreement" in confusion_reasons else 0.0

    confusion_score = round(
        0.35 * conf_component
        + 0.25 * margin_component
        + 0.2 * vote_component
        + 0.1 * format_component
        + 0.1 * verifier_component,
        4,
    )

    hard_confusion = (
        "format_uncertainty" in confusion_reasons
        or (primary_confidence is not None and primary_confidence < args.cascade_hard_confidence)
        or (primary_margin is not None and primary_margin < args.cascade_hard_margin)
        or ("verifier_disagreement" in confusion_reasons)
        or (fallback_choice is None)
    )
    soft_confusion = confusion_score >= args.cascade_confusion_threshold
    should_escalate = hard_confusion or soft_confusion

    max_allowed_escalations = int(question_index_1based * args.cascade_max_escalation_rate + 0.999999)
    budget_allowed = escalations_so_far < max_allowed_escalations

    escalated = False
    final_text = diagnostic_text
    final_choice = fallback_choice
    final_model = primary_model
    escalation_reason = None

    if should_escalate and budget_allowed:
        escalated = True
        final_model = cascade_model
        escalation_reason = ",".join(confusion_reasons) if confusion_reasons else "soft_confusion"

        cascade_text, cascade_usage = call_model(
            client=client,
            model_name=cascade_model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            max_retries=max_retries,
        )
        for key in total_usage:
            total_usage[key] += cascade_usage[key]
            cascade_usage_only[key] += cascade_usage[key]

        cascade_choice = extract_choice(cascade_text, choices)
        if cascade_choice:
            final_choice = cascade_choice
        final_text = cascade_text
    elif should_escalate and not budget_allowed:
        escalation_reason = "budget_blocked"

    return {
        "final_choice": final_choice,
        "response_text": final_text,
        "usage": total_usage,
        "usage_primary": primary_usage,
        "usage_cascade": cascade_usage_only,
        "escalated": escalated,
        "final_model": final_model,
        "escalation_reason": escalation_reason,
        "confusion_score": confusion_score,
        "primary_choice": primary_choice,
        "primary_confidence": primary_confidence,
        "primary_margin": primary_margin,
        "vote_share": vote_share,
        "verifier_choice": verifier_choice,
    }


def evaluate_task(
    client,
    csv_file: Path,
    model_name: str,
    cascade_model_name: str,
    enable_cascade: bool,
    mitre_retriever: Optional[MitreKnowledgeBaseRetriever],
    enable_mitre_kb: bool,
    max_questions_per_task: Optional[int],
    max_retries: int,
    request_max_tokens: int,
    temperature: float,
    sleep_between_requests: float,
    args: argparse.Namespace,
) -> tuple[List[Prediction], Dict[str, int]]:
    task_name = csv_file.stem
    predictions: List[Prediction] = []
    usage_totals = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "escalated": 0,
        "confusion_hits": 0,
        "primary_prompt_tokens": 0,
        "primary_completion_tokens": 0,
        "primary_total_tokens": 0,
        "cascade_prompt_tokens": 0,
        "cascade_completion_tokens": 0,
        "cascade_total_tokens": 0,
    }

    with open(csv_file, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if max_questions_per_task is not None and idx >= max_questions_per_task:
                break

            choices = build_answer_choices(row, task_name)
            mitre_context_block = None
            if enable_mitre_kb and mitre_retriever is not None:
                mitre_context_block = mitre_retriever.build_context_block(task_name, row, choices)
            prompt = build_user_prompt(row, choices, mitre_context_block=mitre_context_block)

            response_text = ""
            predicted_choice = None
            usage_counters = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            escalated = False
            escalation_reason = None
            confusion_score = None
            primary_choice = None
            primary_confidence = None
            primary_margin = None
            vote_share = None
            verifier_choice = None
            model_used = model_name

            if enable_cascade:
                cascade = run_cascade_inference(
                    client=client,
                    prompt=prompt,
                    choices=choices,
                    primary_model=model_name,
                    cascade_model=cascade_model_name,
                    max_tokens=request_max_tokens,
                    max_retries=max_retries,
                    temperature=temperature,
                    question_index_1based=idx + 1,
                    escalations_so_far=usage_totals["escalated"],
                    args=args,
                )
                response_text = cascade["response_text"]
                predicted_choice = cascade["final_choice"]
                usage_counters = cascade["usage"]
                usage_primary = cascade["usage_primary"]
                usage_cascade = cascade["usage_cascade"]
                escalated = bool(cascade["escalated"])
                escalation_reason = cascade["escalation_reason"]
                confusion_score = cascade["confusion_score"]
                primary_choice = cascade["primary_choice"]
                primary_confidence = cascade["primary_confidence"]
                primary_margin = cascade["primary_margin"]
                vote_share = cascade["vote_share"]
                verifier_choice = cascade["verifier_choice"]
                model_used = cascade["final_model"]

                if escalated:
                    usage_totals["escalated"] += 1
                if escalation_reason and escalation_reason != "budget_blocked":
                    usage_totals["confusion_hits"] += 1

                usage_totals["primary_prompt_tokens"] += usage_primary["prompt_tokens"]
                usage_totals["primary_completion_tokens"] += usage_primary["completion_tokens"]
                usage_totals["primary_total_tokens"] += usage_primary["total_tokens"]
                usage_totals["cascade_prompt_tokens"] += usage_cascade["prompt_tokens"]
                usage_totals["cascade_completion_tokens"] += usage_cascade["completion_tokens"]
                usage_totals["cascade_total_tokens"] += usage_cascade["total_tokens"]
            else:
                response_text, usage_counters = call_model(
                    client=client,
                    model_name=model_name,
                    prompt=prompt,
                    max_tokens=request_max_tokens,
                    temperature=temperature,
                    max_retries=max_retries,
                )
                predicted_choice = extract_choice(response_text, choices)
                usage_totals["primary_prompt_tokens"] += usage_counters["prompt_tokens"]
                usage_totals["primary_completion_tokens"] += usage_counters["completion_tokens"]
                usage_totals["primary_total_tokens"] += usage_counters["total_tokens"]

            usage_totals["prompt_tokens"] += usage_counters["prompt_tokens"]
            usage_totals["completion_tokens"] += usage_counters["completion_tokens"]
            usage_totals["total_tokens"] += usage_counters["total_tokens"]

            predicted_answer = choices.get(predicted_choice) if predicted_choice else None

            ground_truth = normalize_ground_truth(row.get("Ground Truth", ""), choices)
            is_correct = bool(
                predicted_answer
                and predicted_answer.strip().casefold() == ground_truth.strip().casefold()
            )

            prediction = Prediction(
                task=task_name,
                question_id=(row.get("Question ID") or "").strip(),
                attackseq_id=(row.get("AttackSeq ID") or "").strip(),
                ground_truth=ground_truth,
                predicted_choice=predicted_choice,
                predicted_answer=predicted_answer,
                is_correct=is_correct,
                prompt_tokens=usage_counters["prompt_tokens"],
                completion_tokens=usage_counters["completion_tokens"],
                total_tokens=usage_counters["total_tokens"],
                response_text=response_text,
                model_used=model_used,
                escalated=escalated,
                escalation_reason=escalation_reason,
                confusion_score=confusion_score,
                primary_choice=primary_choice,
                primary_confidence=primary_confidence,
                primary_margin=primary_margin,
                vote_share=vote_share,
                verifier_choice=verifier_choice,
            )
            predictions.append(prediction)

            if len(predictions) % 20 == 0:
                correct_so_far = sum(1 for p in predictions if p.is_correct)
                print(
                    f"[{task_name}] Processed {len(predictions)} questions, "
                    f"accuracy so far: {correct_so_far / len(predictions):.2%}"
                )

            if sleep_between_requests > 0:
                time.sleep(sleep_between_requests)

    return predictions, usage_totals


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value)


def main() -> None:
    run_start_time = time.perf_counter()
    root_dir = Path(__file__).resolve().parent
    load_dotenv_local(root_dir / ".env")
    args = parse_args()
    pricing = resolve_pricing(args.model, args)
    cascade_pricing = resolve_pricing(args.cascade_model, args)

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY not found. Add it to .env in the current working directory."
        )

    dataset_dir: Path = args.dataset_dir
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    csv_files = sorted(dataset_dir.glob("AttackSeq-*.csv"))
    if args.tasks:
        requested = set(args.tasks)
        csv_files = [f for f in csv_files if f.stem in requested or f.name in requested]

    if not csv_files:
        raise ValueError("No task CSV files found. Check --dataset-dir or --tasks.")

    mitre_retriever: Optional[MitreKnowledgeBaseRetriever] = None
    if args.enable_mitre_kb:
        if not args.mitre_kb_path.exists():
            raise FileNotFoundError(f"MITRE KB JSON not found: {args.mitre_kb_path}")
        mitre_retriever = MitreKnowledgeBaseRetriever(
            kb_path=args.mitre_kb_path,
            top_k=args.mitre_top_k,
            max_chars=args.mitre_max_chars,
        )

    if OPENAI_SDK_MODE == "modern":
        client = OpenAI()
    else:
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if os.getenv("OPENAI_BASE_URL"):
            openai.api_base = os.getenv("OPENAI_BASE_URL")
        client = None

    all_predictions: List[Prediction] = []
    per_task_summary: Dict[str, Dict[str, float]] = {}
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0

    for csv_file in csv_files:
        task_name = csv_file.stem
        if args.enable_cascade:
            print(
                f"Evaluating {task_name} with primary={args.model}, "
                f"cascade={args.cascade_model}..."
            )
        else:
            print(f"Evaluating {task_name} with model {args.model}...")
        task_predictions, task_usage = evaluate_task(
            client=client,
            csv_file=csv_file,
            model_name=args.model,
            cascade_model_name=args.cascade_model,
            enable_cascade=args.enable_cascade,
            mitre_retriever=mitre_retriever,
            enable_mitre_kb=args.enable_mitre_kb,
            max_questions_per_task=args.max_questions_per_task,
            max_retries=args.max_retries,
            request_max_tokens=args.request_max_tokens,
            temperature=args.temperature,
            sleep_between_requests=args.sleep_between_requests,
            args=args,
        )

        total = len(task_predictions)
        correct = sum(1 for p in task_predictions if p.is_correct)
        accuracy = (correct / total) if total else 0.0
        task_prompt_tokens = task_usage["prompt_tokens"]
        task_completion_tokens = task_usage["completion_tokens"]
        task_total_tokens = task_usage["total_tokens"]
        if args.enable_cascade:
            task_estimated_cost = estimate_mixed_cost_usd(
                primary_prompt_tokens=int(task_usage.get("primary_prompt_tokens", 0)),
                primary_completion_tokens=int(task_usage.get("primary_completion_tokens", 0)),
                primary_pricing=pricing,
                cascade_prompt_tokens=int(task_usage.get("cascade_prompt_tokens", 0)),
                cascade_completion_tokens=int(task_usage.get("cascade_completion_tokens", 0)),
                cascade_pricing=cascade_pricing,
            )
        else:
            task_estimated_cost = estimate_cost_usd(
                prompt_tokens=task_prompt_tokens,
                completion_tokens=task_completion_tokens,
                input_cost_per_1m=pricing["input"],
                output_cost_per_1m=pricing["output"],
            )
        per_task_summary[task_name] = {
            "total": total,
            "correct": correct,
            "accuracy": accuracy,
            "prompt_tokens": task_prompt_tokens,
            "completion_tokens": task_completion_tokens,
            "total_tokens": task_total_tokens,
            "estimated_cost_usd": task_estimated_cost if task_estimated_cost is not None else -1.0,
            "escalated": int(task_usage.get("escalated", 0)),
            "confusion_hits": int(task_usage.get("confusion_hits", 0)),
            "primary_prompt_tokens": int(task_usage.get("primary_prompt_tokens", 0)),
            "primary_completion_tokens": int(task_usage.get("primary_completion_tokens", 0)),
            "cascade_prompt_tokens": int(task_usage.get("cascade_prompt_tokens", 0)),
            "cascade_completion_tokens": int(task_usage.get("cascade_completion_tokens", 0)),
        }
        total_prompt_tokens += task_prompt_tokens
        total_completion_tokens += task_completion_tokens
        total_tokens += task_total_tokens
        all_predictions.extend(task_predictions)
        print(f"Finished {task_name}: {correct}/{total} ({accuracy:.2%})")

    overall_total = len(all_predictions)
    overall_correct = sum(1 for p in all_predictions if p.is_correct)
    overall_accuracy = (overall_correct / overall_total) if overall_total else 0.0
    macro_accuracy = (
        sum(task["accuracy"] for task in per_task_summary.values()) / len(per_task_summary)
        if per_task_summary
        else 0.0
    )
    total_primary_prompt = int(sum(task.get("primary_prompt_tokens", 0) for task in per_task_summary.values()))
    total_primary_completion = int(
        sum(task.get("primary_completion_tokens", 0) for task in per_task_summary.values())
    )
    total_cascade_prompt = int(sum(task.get("cascade_prompt_tokens", 0) for task in per_task_summary.values()))
    total_cascade_completion = int(
        sum(task.get("cascade_completion_tokens", 0) for task in per_task_summary.values())
    )

    if args.enable_cascade:
        estimated_cost_usd = estimate_mixed_cost_usd(
            primary_prompt_tokens=total_primary_prompt,
            primary_completion_tokens=total_primary_completion,
            primary_pricing=pricing,
            cascade_prompt_tokens=total_cascade_prompt,
            cascade_completion_tokens=total_cascade_completion,
            cascade_pricing=cascade_pricing,
        )
    else:
        estimated_cost_usd = estimate_cost_usd(
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            input_cost_per_1m=pricing["input"],
            output_cost_per_1m=pricing["output"],
        )
    run_duration_seconds = time.perf_counter() - run_start_time

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model_slug = safe_name(args.model)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / f"predictions_{model_slug}_{timestamp}.jsonl"
    summary_path = args.output_dir / f"summary_{model_slug}_{timestamp}.json"

    with open(predictions_path, "w", encoding="utf-8") as f:
        for item in all_predictions:
            f.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")

    summary = {
        "model": args.model,
        "cascade": {
            "enabled": args.enable_cascade,
            "cascade_model": args.cascade_model,
            "max_escalation_rate": args.cascade_max_escalation_rate,
            "min_confidence": args.cascade_min_confidence,
            "min_margin": args.cascade_min_margin,
            "hard_confidence": args.cascade_hard_confidence,
            "hard_margin": args.cascade_hard_margin,
            "self_consistency_votes": args.cascade_self_consistency_votes,
            "self_consistency_temperature": args.cascade_self_consistency_temperature,
            "min_vote_share": args.cascade_min_vote_share,
            "confusion_threshold": args.cascade_confusion_threshold,
            "enable_verifier": args.cascade_enable_verifier,
            "diagnostic_max_tokens": args.cascade_diagnostic_max_tokens,
        },
        "mitre_kb": {
            "enabled": args.enable_mitre_kb,
            "path": str(args.mitre_kb_path.resolve()) if args.enable_mitre_kb else None,
            "top_k": args.mitre_top_k if args.enable_mitre_kb else None,
            "max_chars": args.mitre_max_chars if args.enable_mitre_kb else None,
        },
        "dataset_dir": str(dataset_dir.resolve()),
        "tasks": [f.stem for f in csv_files],
        "max_questions_per_task": args.max_questions_per_task,
        "runtime": {
            "seconds": round(run_duration_seconds, 3),
            "formatted": format_duration(run_duration_seconds),
        },
        "token_usage": {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens,
            "primary_prompt_tokens": total_primary_prompt,
            "primary_completion_tokens": total_primary_completion,
            "cascade_prompt_tokens": total_cascade_prompt,
            "cascade_completion_tokens": total_cascade_completion,
        },
        "cost": {
            "estimated_cost_usd": estimated_cost_usd,
            "input_cost_per_1m": pricing["input"],
            "output_cost_per_1m": pricing["output"],
            "pricing_source": pricing["source"],
            "cascade_input_cost_per_1m": cascade_pricing["input"] if args.enable_cascade else None,
            "cascade_output_cost_per_1m": cascade_pricing["output"] if args.enable_cascade else None,
            "cascade_pricing_source": cascade_pricing["source"] if args.enable_cascade else None,
        },
        "overall": {
            "total": overall_total,
            "correct": overall_correct,
            "accuracy": overall_accuracy,
            "macro_accuracy": macro_accuracy,
            "escalated": int(sum(task["escalated"] for task in per_task_summary.values())),
            "confusion_hits": int(sum(task["confusion_hits"] for task in per_task_summary.values())),
        },
        "per_task": per_task_summary,
        "artifacts": {
            "predictions_jsonl": str(predictions_path.resolve()),
            "summary_json": str(summary_path.resolve()),
        },
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== AttackSeqBench Benchmark Summary ===")
    for task_name, task in per_task_summary.items():
        per_task_cost_text = "N/A"
        if task["estimated_cost_usd"] >= 0:
            per_task_cost_text = f"${task['estimated_cost_usd']:.6f}"
        extra = ""
        if args.enable_cascade:
            extra = (
                f" escalated={int(task['escalated'])}"
                f" confusion_hits={int(task['confusion_hits'])}"
            )
        print(
            f"{task_name:24} "
            f"{int(task['correct'])}/{int(task['total'])} "
            f"({task['accuracy']:.2%}) "
            f"tokens={int(task['total_tokens'])} "
            f"est_cost={per_task_cost_text}{extra}"
        )
    print(
        f"Overall: {overall_correct}/{overall_total} ({overall_accuracy:.2%}), "
        f"Macro Accuracy: {macro_accuracy:.2%}"
    )
    if args.enable_cascade and overall_total:
        overall_escalated = summary["overall"]["escalated"]
        print(
            f"Cascade escalation: {overall_escalated}/{overall_total} "
            f"({overall_escalated / overall_total:.2%})"
        )
    print(f"Runtime: {format_duration(run_duration_seconds)} ({run_duration_seconds:.2f}s)")
    print(
        f"Token Usage: prompt={total_prompt_tokens}, "
        f"completion={total_completion_tokens}, total={total_tokens}"
    )
    if estimated_cost_usd is not None:
        print(
            f"Estimated Cost (USD): ${estimated_cost_usd:.6f} "
            f"(pricing source: {pricing['source']}, "
            f"input ${pricing['input']}/1M, output ${pricing['output']}/1M)"
        )
    else:
        print(
            "Estimated Cost (USD): N/A "
            "(no pricing known for this model; use --input-cost-per-1m and --output-cost-per-1m)"
        )
    print(f"Predictions saved to: {predictions_path}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
