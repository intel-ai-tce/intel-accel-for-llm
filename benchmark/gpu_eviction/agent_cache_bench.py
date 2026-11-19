#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Repeatable OpenAI-compatible KV-cache reuse and eviction-pressure benchmark.

No assumptions are made about model-specific cache-hit telemetry. The report
separates observed latency from cache-hit evidence supplied by the server.
"""

import argparse
import concurrent.futures
import json
import math
import statistics
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class Sample:
    phase: str
    session: int
    turn: int
    elapsed_ms: float
    ttft_ms: float | None
    completion_tokens: int
    prompt_tokens: int | None
    cached_tokens: int | None
    status: str
    error: str = ""


def request_json(url: str, payload: dict, timeout: float, api_key: str) -> dict:
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, body, headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def completion(url: str, model: str, prompt: str, output_tokens: int,
               timeout: float, api_key: str, phase: str, session: int,
               turn: int) -> tuple[Sample, str]:
    payload = {
        "model": model, "prompt": prompt, "max_tokens": output_tokens,
        "temperature": 0, "ignore_eos": True, "stream": True,
        "stream_options": {"include_usage": True},
    }
    start = time.perf_counter()
    first_token: float | None = None
    content: list[str] = []
    usage: dict[str, Any] = {}
    try:
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        req = urllib.request.Request(url, json.dumps(payload).encode(), headers, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as response:
            for raw in response:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                chunk = line[5:].strip()
                if chunk == "[DONE]":
                    break
                event = json.loads(chunk)
                if event.get("error"):
                    raise RuntimeError(str(event["error"]))
                if isinstance(event.get("usage"), dict):
                    usage = event["usage"]
                for choice in event.get("choices") or []:
                    token = choice.get("text", "")
                    if token:
                        first_token = first_token or time.perf_counter()
                        content.append(token)
        end = time.perf_counter()
        details = usage.get("prompt_tokens_details") or {}
        cached = details.get("cached_tokens")
        prompt_count = usage.get("prompt_tokens")
        generated = usage.get("completion_tokens")
        sample = Sample(
            phase, session, turn, (end - start) * 1000,
            ((first_token - start) * 1000 if first_token else None),
            int(generated if generated is not None else 0),
            int(prompt_count) if prompt_count is not None else None,
            int(cached) if cached is not None else None, "ok",
        )
        return sample, "".join(content)
    except (OSError, ValueError, RuntimeError) as exc:
        end = time.perf_counter()
        return Sample(phase, session, turn, (end - start) * 1000, None,
                      0, None, None, "error", str(exc)), ""


def fragment(label: str, words: int) -> str:
    # Long, stable prefixes. Distinct labels prevent accidental cross-session reuse.
    phrase = f"[{label}] Analyze the evidence, constraints, and implications carefully. "
    return (phrase * math.ceil(words / len(phrase.split())))[: max(1, words) * 90]


def make_prefix(label: str, words: int) -> str:
    phrase = f"Case {label}: review the system context and preserve all prior facts. "
    return (phrase * math.ceil(words / len(phrase.split()))) + "\n"


def summary(samples: list[Sample]) -> dict:
    by_phase: dict[str, list[Sample]] = {}
    for sample in samples:
        by_phase.setdefault(sample.phase, []).append(sample)
    result = {}
    for phase, data in by_phase.items():
        successful = [s for s in data if s.status == "ok"]
        ttfts = sorted(s.ttft_ms for s in successful if s.ttft_ms is not None)
        latencies = [s.elapsed_ms for s in successful]
        cached = [s.cached_tokens for s in successful if s.cached_tokens is not None]
        result[phase] = {
            "requests": len(data), "success": len(successful),
            "errors": len(data) - len(successful),
            "ttft_p50_ms": round(statistics.median(ttfts), 2) if ttfts else None,
            "ttft_p99_ms": round(ttfts[min(len(ttfts) - 1, math.ceil(.99 * len(ttfts)) - 1)], 2) if ttfts else None,
            "mean_latency_ms": round(statistics.mean(latencies), 2) if latencies else None,
            "cached_tokens_available": len(cached),
            "cached_tokens_mean": round(statistics.mean(cached), 1) if cached else None,
        }
    return result


def run_eviction(args: argparse.Namespace) -> tuple[list[Sample], dict]:
    url = f"{args.endpoint.rstrip('/')}/v1/completions"
    samples: list[Sample] = []
    prompt_a = make_prefix("TARGET_A", args.prefix_words) + "\nQuestion: Summarize the case.\nAnswer:"

    def send(phase: str, prompt: str, index: int):
        sample, _ = completion(url, args.model, prompt, args.output_tokens,
                               args.timeout, args.api_key, phase, index, 0)
        samples.append(sample)
        if sample.status != "ok":
            raise RuntimeError(f"{phase} request {index}: {sample.error}")
        print(f"{phase:>14} #{index:<3} TTFT={sample.ttft_ms}ms cached={sample.cached_tokens}", flush=True)
        return sample

    cold = send("target_cold", prompt_a, 0)
    warm = send("target_warm", prompt_a, 0)
    for i in range(args.pressure_prompts):
        pressure = make_prefix(f"PRESSURE_{i:05d}", args.prefix_words)
        send("cache_pressure", pressure + "\nQuestion: Summarize the case.\nAnswer:", i)
    if args.settle_seconds:
        time.sleep(args.settle_seconds)
    resumed = send("target_resume", prompt_a, 0)
    verification = "unverified"
    if None not in (cold.cached_tokens, warm.cached_tokens, resumed.cached_tokens):
        threshold = args.min_cached_tokens
        if warm.cached_tokens >= threshold and resumed.cached_tokens < threshold:
            verification = "gpu_eviction_evidenced_by_cached_tokens"
        elif warm.cached_tokens < threshold:
            verification = "gpu_warm_hit_not_observed"
        else:
            verification = "gpu_hit_still_present_or_external_hits_combined"
    return samples, {"gpu_eviction_verification": verification,
                     "cached_tokens_semantics": "Server-specific; external hits may be included or absent",
                     "target_cold_ttft_ms": cold.ttft_ms,
                     "target_warm_ttft_ms": warm.ttft_ms,
                     "target_resume_ttft_ms": resumed.ttft_ms}


def run_agents(args: argparse.Namespace) -> tuple[list[Sample], dict]:
    url = f"{args.endpoint.rstrip('/')}/v1/completions"
    histories = [make_prefix(f"AGENT_{i:04d}", args.prefix_words)
                 for i in range(args.sessions)]
    samples: list[Sample] = []
    # Complete each round before advancing: increasing reuse distance with sessions.
    for turn in range(args.turns):
        def one(session: int) -> tuple[int, Sample, str]:
            prompt = histories[session] + f"\nUser turn {turn + 1}: Provide the next grounded conclusion.\nAssistant:"
            result, output = completion(url, args.model, prompt, args.output_tokens,
                                        args.timeout, args.api_key, "agent_turn", session, turn + 1)
            return session, result, output

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            results = list(executor.map(one, range(args.sessions)))
        for session, sample, output in results:
            samples.append(sample)
            if sample.status != "ok":
                raise RuntimeError(f"agent {session} turn {turn + 1}: {sample.error}")
            # Exact previous prompt plus actual output becomes the next turn's prefix.
            histories[session] += (f"\nUser turn {turn + 1}: Provide the next grounded conclusion."
                                   f"\nAssistant:{output}\n")
        ttfts = [s.ttft_ms for _, s, _ in results if s.ttft_ms is not None]
        print(f"turn={turn + 1} completed={len(results)} median_ttft_ms="
              f"{round(statistics.median(ttfts), 2) if ttfts else 'NA'}", flush=True)
        if args.think_seconds and turn + 1 < args.turns:
            time.sleep(args.think_seconds)
    return samples, {"history_policy": "exact prior prompt + generated completion",
                     "scheduling": "round-robin turns; concurrency within each turn"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["eviction", "agents"])
    parser.add_argument("--endpoint", default="http://localhost:8000")
    parser.add_argument("--model", default="Qwen/Qwen2.5-72B-Instruct")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--prefix-words", type=int, default=4500,
                        help="Approximate prefix words, not exact tokens")
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--output", default="benchmark/gpu_eviction/results.json")
    parser.add_argument("--label", default="unlabeled", help="e.g. gpu-only or iaxl-dsa")
    parser.add_argument("--pressure-prompts", type=int, default=24)
    parser.add_argument("--settle-seconds", type=float, default=5)
    parser.add_argument("--min-cached-tokens", type=int, default=128)
    parser.add_argument("--sessions", type=int, default=32)
    parser.add_argument("--turns", type=int, default=6)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--think-seconds", type=float, default=0)
    args = parser.parse_args()
    if min(args.prefix_words, args.output_tokens, args.sessions, args.turns,
           args.concurrency, args.min_cached_tokens) <= 0 or args.pressure_prompts < 0:
        parser.error("sizes/counts must be positive; pressure-prompts may be zero")
    if args.timeout <= 0 or args.settle_seconds < 0 or args.think_seconds < 0:
        parser.error("timeout must be positive and delays nonnegative")
    return args


def main() -> int:
    args = parse_args()
    began = time.time()
    try:
        samples, evidence = (run_eviction(args) if args.mode == "eviction"
                             else run_agents(args))
    except (RuntimeError, KeyboardInterrupt) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    result = {
        "schema_version": 1, "mode": args.mode, "label": args.label,
        "model": args.model, "endpoint": args.endpoint,
        "started_unix": began,
        "parameters": {k: v for k, v in vars(args).items() if k != "api_key"},
        "evidence": evidence, "summary": summary(samples),
        "samples": [asdict(sample) for sample in samples],
        "notes": [
            "Prefix length is approximate; inspect returned prompt_tokens for actual size.",
            "TTFT is client-observed first streamed token, including HTTP overhead.",
            "Latency alone cannot prove a host KV hit. Correlate IAXL connector logs/metrics.",
            "Each label/configuration requires a separately initialized server/cache state.",
        ],
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    print(f"Report: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
