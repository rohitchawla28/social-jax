# CLAUDE.md

## Project goal
This repo is the main SocialJax codebase. Current goal: port the HMASD method from the reference PyTorch implementation into JAX/Flax inside this repo as another algorithm, then later add the same style of metrics, logging, and evaluation chunking used in the existing IPPO/MAPPO baselines (as seen in cleanup and coins).

## Source-of-truth rules
- Treat this SocialJax repo as the primary codebase.
- Treat the HMASD PyTorch code under `NeurIPS2023_HMASD_code/` as reference-only.
- Do not restructure the repo unless there is a strong reason.
- Do not replace existing SocialJax training/eval patterns with a brand new framework unless explicitly asked.
- We want consistency with existing SocialJax baselines/code over clever abstractions.

## Dependency rules
- Preserve the current SocialJax dependency stack.
- Do not modify repo-wide package versions, environment files, or install scripts unless necessary and clearly justified.
- If a dependency change seems needed, explain why before making it.

## Implementation preferences
- Reuse existing baseline structure and utilities where reasonable.
- Prefer simple, readable code over heavy abstraction.
- Keep new files/modules aligned with the style of the current JAX baselines.
- Avoid introducing unnecessary wrappers, helper layers, or configuration systems.
- Avoid unnecessary bloat code, this isn't a production system, this is just a research project, where we really need reproducibility and baseline experiments
- Make the smallest viable change that moves the task (within the milestone) forward.

## Workflow
For larger tasks:
1. Inspect all relevant files first.
2. Propose a short implementation plan.
3. Implement one milestone at a time.
4. Run verification checks before declaring success.
5. Summarize what changed and any open issues.

Do not try to do the entire project in one giant step.

## Testing and verification
Before calling a milestone complete, verify with the smallest useful checks available.

Preferred early checks:
- imports succeed
- model init/apply works on dummy inputs
- output shapes are correct
- one loss computation runs
- one optimizer step runs
- no NaNs in the basic path
- rollout/eval path runs if relevant

For later integration work, also verify:
- expected logging keys exist
- evaluation chunking works
- metric outputs follow existing baseline conventions

## Logging / metrics preferences
- Match the existing IPPO/MAPPO logging style (from coins & cleanup envs) where possible.
- Want consistency in metric naming and aggregation.
- Do not add a large amount of instrumentation until the core implementation is working.
- Reuse existing evaluation/logging helpers when possible.

## Git / editing rules
- Make focused changes tied to the current milestone.
- Do not make unrelated cleanup edits.
- Do not commit automatically unless explicitly asked.
- If asked to commit, only commit after verification checks pass.

## Communication preferences
- Be explicit about assumptions.
- If adapting the HMASD method because of SocialJax environment or architecture differences, say so clearly.
- When multiple implementation options exist, prefer the one that fits the current repo patterns best.
- Keep explanations practical, moderately concise meaning explain when there are more complex ideas or decisions involved.

## Important context for this repo
- SocialJax env/training code is already working and should be preserved.
- Existing IPPO/MAPPO files contain the desired style for metrics and eval chunking.
- HMASD reference code is useful for algorithmic guidance, not as the execution stack to preserve exactly.
- Can use HMASD reference code (their runners) to see how they ran their algorithm and see how that can be aligned with the SocialJax algorithm styles