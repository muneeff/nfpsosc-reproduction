# V2 Internal Ablation Corpus Closure

**Closure date:** 2026-08-15  
**Status:** `CLOSED_PRE_INFERENCE`

## Frozen execution boundary

- Execution code commit: `aa936ab5f353ee1063d69a0a0484edf1a87da273`
- Execution-freeze tag: `v2-ablation-execution-freeze-2026-08-14`
- Execution-freeze SHA-256: `49a8d5d3277685f712130446db57b9271b8945154c352a3480c9ce59d4115359`
- Manifest SHA-256: `6dedd4dd75c0953d617679226800daec5e072fb45f69af0ed3667f94a6ff376b`
- Protocol fingerprint: `890c26e2b306e544a67e55e24f4f76d18bc021eff0ded8d8d8e0610ad851d6dc`

## New ablation corpus

- Result files: **5,250**
- Successful: **5,250**
- Failed: **0**
- Pending: **0**
- Result corpus SHA-256: `3628d71f833bd50ae9c5ff8799568e159bb89afa525e6d3bc5bafc77bc44a45b`
- Per-file-hash aggregate SHA-256: `82a4179cb2c3e9dd51f1f655273dc71045c6242fb60fd4676dddf7fd9ad26551`

## Reused PC_NFPSO reference corpus

- Frozen reference rows: **1,410**
- Closed external Final aggregate SHA-256: `393b7df1fca6d6dc2e57b60b06d6ccca7f15217e3a8c351b9427346829da299b`

## Execution

- Workers: **1**
- Threads per worker: **1**
- Max in flight: **2**
- Wall time: **6376.283389 s**
- Stopped after failure: **False**

## Operational incidents

1. The first external launcher omitted the Windows multiprocessing `__main__`
   guard and failed before any result checkpoint was written. The replacement
   launcher changed only the external launch idiom; no frozen scientific code,
   protocol, manifest, execution metadata, worker policy, or task identity was
   changed.
2. After all 5,250 results had completed successfully, a Git-cleanliness
   assertion was triggered by an untracked hidden Microsoft Word temporary file
   (`~WRL2493.tmp`). Closing Word removed the file automatically. No tracked or
   staged repository content changed.

## Inference gate

No statistical ablation inference was run before this closure. No scientific
outcome was recomputed during closure. Statistical analysis may begin only
after these closure artifacts are committed and tagged.
