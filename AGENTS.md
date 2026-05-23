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
- `dev_remodel` is the **source of truth** branch.
- All work must be done on a `refactor_remodel/<feature>` branch.
- Changes go to `dev_remodel` only through a **PR from the `refactor_remodel/` branch**.
- Never commit directly to `dev_remodel`.
