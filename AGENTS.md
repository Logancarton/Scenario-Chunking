# Instructions for coding and research agents

**Required first read:** [GOAL.md](GOAL.md). This is the authoritative long-term Sonara objective and the planning contract. Re-read it when starting a new task, after context compaction, and before any large architectural decision. The [research journal](docs/COGNITIVE_RESEARCH_JOURNAL.md) and verified source/tests determine *implemented truth*, not the goal document.

## Repository and scope

This repository is **Scenario-Chunking**, an isolated cognitive research sandbox. It is **not** Sonara production. The long-term direction is self-directed, persistent learning from real authorized online/offline resources, grounded language/code/process understanding, transferable prediction, and controlled improvement of the learning mechanisms themselves. Do not state those future capabilities already exist.

Before editing:
- Inspect the complete relevant receiving-owner signal path, call sites, state transitions, test harness, and existing behavior.
- Respect single-owner cognition: no duplicate planners, selectors, memory authorities, learning systems, or monolithic runtimes.
- Treat experiments as evidence and proposed transplants as *unverified* until they reach the genuine production path and show observable effects.
- Keep legacy Sonara architecture quarantined; do not treat the research harness as the production runtime.

## Planning protocol

For every substantive task, state: **STATUS** (current verified evidence and gap), **NEXT** (one concrete action), **EXPECTED** (observable behavior, comparison/baselines, integration tests, full-suite/gates, compilation, diff, documentation and git outcomes).

Use `GOAL.md` to choose the objective, not to justify a predetermined implementation. Evaluate prediction *and* decision reward, unseen transfer, computational cost, contamination/leakage, uncertainty, restart persistence, and effects on the right learning/memory owner. Negative findings are valuable and must remain recorded.

Before marking completion: focused tests, live integration/scenario, full suite, research/engine gate if applicable, compile, `git diff --check`, scoped diff/status review, documentation truth synchronization, and a focused reversible commit/push. Never equate a green unit-test suite with proven cognitive improvement. Do not use `git add .`, `git clean`, or `git reset --hard`; stage exact paths only.

Agent-specific files such as `CLAUDE.md` and `.github/copilot-instructions.md` point back here; do not fork the goal into competing instructions.
