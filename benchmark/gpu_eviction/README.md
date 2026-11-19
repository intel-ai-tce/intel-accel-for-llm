# GPU prefix-cache eviction and multi-turn agent benchmark

This benchmark tests **GPU KV-prefix reuse under capacity pressure** and whether
IAXL can restore a matching prefix from the host cache. It is separate from
`doc/design/eviction.md`, which describes eviction **from the IAXL DDR pool**.

## Prerequisites

Run on the `GPU_Eviction` branch in the development container documented in
`AGENTS.md`. Start the serving process separately and use the *same* model,
TP, cache dtype, workload, output length, and GPU cache size for comparisons.

The repo's `examples/kvshrink-vllm-serve.sh` defaults to
`--no-enable-prefix-caching`. **Override this when benchmarking a combined
GPU + external-cache configuration**, rather than interpreting disabled GPU
prefix caching as GPU eviction. The script honors `VLLM_SERVE_CMDLINE_ARGS`;
when using it, supply the complete desired set of serving flags.

Example **GPU-only** vLLM invocation in the development container:

```bash
vllm serve Qwen/Qwen2.5-72B-Instruct \
  --tensor-parallel-size 8 --max-model-len 32768 \
  --enable-prefix-caching --gpu-memory-utilization 0.8 \
  --num-gpu-blocks-override 8192 --port 8000
```

Example **GPU + IAXL** launch (with the same cache-block override):

```bash
MODEL=Qwen/Qwen2.5-72B-Instruct TP_SIZE=8 \
VLLM_SERVE_CMDLINE_ARGS='--trust-remote-code --max-model-len 32768 --port 8000 --enable-prefix-caching --gpu-memory-utilization 0.8 --num-gpu-blocks-override 8192' \
  bash examples/kvshrink-vllm-serve.sh
```

Set the existing IAXL environment flags (`IAXL_DSA_GD_ENABLE`,
`IAXL_QAT_ZIP_ENABLE`, etc.) via `setvars.sh` or the environment, and check the
actual server startup log before comparing runs. `8192` is an **example**, not a
validated safe size or a guarantee of eviction on Qwen2.5-72B. Tune for your
hardware, active request count, and model KV geometry.

## Single-prefix warm / evict / resume test

```bash
python3 benchmark/gpu_eviction/agent_cache_bench.py eviction \
  --model Qwen/Qwen2.5-72B-Instruct \
  --prefix-words 4500 --pressure-prompts 24 \
  --label gpu-only --output /_data/gpu-only-eviction.json
```

The harness runs target A cold, repeats A (warm), issues distinct long prompts
that consume GPU KV blocks, then repeats A again (resume). Increase
`--pressure-prompts` or decrease the GPU cache-block allocation until server
telemetry confirms that A was evicted.

**Validation requirements:**

1. Verify that A's warm request hits the native GPU prefix cache.
2. Verify that the GPU-only resumed request recomputes the prefix after pressure.
3. Repeat on a **fresh** IAXL server; verify the target was saved to host and
   its external KV blocks were loaded on resume (connector log lines such as
   `externally-cached tokens: N` with N > 0, plus backend counters if available).
4. Confirm the IAXL host DDR pool has not itself evicted A. Wait long enough
   for asynchronous writes to finish, and keep its cache budget sufficient.
5. Compare TTFT distributions across **multiple independent trials**. Single
   samples are diagnostic only, not reliable performance results.

Cached-token accounting varies by vLLM version. The harness reports an
`unverified` state if the streaming API does not return `cached_tokens`, and
its non-`unverified` indicators still require independent GPU/host evidence.
GPU and external cache hits must not be conflated.

## Multi-turn agent workload

```bash
python3 benchmark/gpu_eviction/agent_cache_bench.py agents \
  --model Qwen/Qwen2.5-72B-Instruct \
  --sessions 32 --turns 6 --concurrency 4 \
  --prefix-words 4500 --output /_data/iaxl-agents.json --label iaxl-dsa
```

Sessions run in turn-based rounds to vary reuse distance. Each subsequent
request preserves the exact preceding prompt and generated response, adding
one new turn. Compare a fresh GPU-only server to GPU+IAXL (DSA-only and
DSA+QAT) at the same configuration. This is a *synthetic agent-conversation*
benchmark; there are no live tool calls.

Output JSON contains per-request TTFT, latency, token counts (when reported),
phase summaries, parameters, and evidence limitations. No artificial speedup
or host-cache-hit claim is made from timing alone.

## Limitations and interpretation

- A longer multi-turn conversation does not inherently lower native GPU hit
  rates. **Working-set pressure and reuse distance** are the relevant factors.
- Reused prefixes must be token-identical. The harness uses text completions
  with zero temperature, not chat-template-specific messages.
- `--prefix-words` is not a precise token count; inspect `prompt_tokens` in
  result JSON. For exact 8K input comparisons, adjust using the model tokenizer.
- The server's `stream_options.include_usage` and `ignore_eos` should be
  supported by the deployed vLLM version. Some servers omit cached-token data.
- A reported native cached-token count may not distinguish GPU and host tokens.
  Verify with vLLM metrics and the KVShrink logs.
- Record server version, GPUs, memory settings, KV cache dtype, connector
  configuration, hit counts, and the IAXL DDR eviction watermarks with results.
