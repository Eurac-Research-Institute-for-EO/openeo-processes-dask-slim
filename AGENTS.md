# Agent Instructions

## Environment
- Always use the micromamba environment `slim_dataset` for all commands (e.g. `micromamba run -n slim_dataset <command>`).

## Session Start
- Read `plan_document.md` and `dev_remodel_review.md` for full context at the beginning of every session.

## Approval Workflow
- When I say **"I approve"**, it means I approve the current changes. You must then:
  1. Run `pre-commit run --all` and fix any issues.
  2. Commit the changes with a descriptive message.
  3. Push the current branch.

## Branch Strategy
- `dev_remodel` is the **final source of truth** branch.
- `temp-source-of-truth` is a **temporary integration branch** created from `dev_remodel` for batch-fixing P0-P3 issues.
- All work must be done on a `refactor_remodel/<feature>` branch.
- Each feature branch is merged into `temp-source-of-truth` via a PR.
- After all issues are fixed, a final PR goes from `temp-source-of-truth` to `dev_remodel`.
- Never commit directly to `dev_remodel`.
