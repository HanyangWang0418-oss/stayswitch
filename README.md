# StaySwitch: cache-aware step-level model routing for LLM agents

Research project on routing each LLM call inside an agent trajectory to the cheapest sufficient model,
treating prompt-cache state and switching cost as part of the decision.

- [`proposal.md`](proposal.md) — research plan, related work map, and running notes (in Chinese)
- [`stayswitch/`](stayswitch/) — LiteLLM-proxy router, replay-based forking, cache-aware accounting, Harbor/Terminal-Bench and SWE-bench tooling (see its README)
