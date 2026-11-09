# Decision log

| ID | Decision | Reason / evidence | Status |
|---|---|---|---|
| 001 | Fixed CortexLab, local open-source-ready delivery | Explicit fixed directive; no remote publication | accepted |
| 002 | Original small trained Torch decoder | Offline provenance, audited capture, measurable learning; avoid unverified model weights | accepted; quality pending |
| 003 | Per-head attention + post-block full residual | Exact numeric inspection and causal-mask tests; no reasoning attribution | accepted |
| 004 | Complete HTTP recordings and replay | Small bounded requests; preserve all frames; WebSockets unnecessary for first slice | accepted |
| 005 | Direct Three/WebGL2 | Established APIs, no need React or GPU compute; official docs reviewed | accepted; dependency access risk |
| 006 | Original MIT source, no upstream code copying | 0xLane AGPL concepts only; separate third-party notices | accepted |
| 007 | Loopback, fixed checkpoint, bounded inputs | Local tool; remote auth/rate limiting out of scope | accepted |
| 008 | Exact recorded values, separately labeled visual transitions | Prevent animation from masquerading as measured computation | accepted |
| 009 | Native ESM JavaScript + strict numerical contract; no build | Removes frontend installation dependency while preserving independent modules and11 contract tests | implemented |
| 010 | Vendor official tagged Three r170 source with MIT/provenance | Network denies npm/browser downloads; raw research-source retrieval worked. Parsing and actual shader/render checks pass; upstream byte identity unverified | implemented; release follow-up |
| 011 | Compact JSON export | Maximum-context pretty JSON exceeded import cap; compact3.43MB round trip verified | implemented/tested |
| 012 | Cross-runtime tensor tolerance for stored-demo re-inference | Exact playback differs from bitwise re-execution; CPU/BLAS roundoff must not produce false CI failures | implemented/tested |

All role reports reviewed: A/H `context-report`, `opensource-report`; B/C `research-report`, `ai-report`; D/E/F/G `graphics-report`, `product-report`, `architecture-report`, `qa-report`. Contradictions resolved: live inference is prior art; WebGPU is not dismissed as universally experimental; latest-only world telemetry must not drop durable model recordings.
