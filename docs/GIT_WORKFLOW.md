# Git workflow (commit history is graded)

1. One shared repo; the coordinator (M10) creates it and adds all members.
2. Never commit directly to `main`. Branch: `feature/<mN>-<short-topic>`.
3. Commit small and often, with your own account, for work you actually did.
   Minimum: at least one meaningful commit per member per week (aim for 2-3).
4. Message style: `<area>: <what changed>`, e.g. `data: parse monthly Excel sheets`.
5. Open a Pull Request; at least one other member reviews and merges.
6. Do not commit: data files, model weights, secrets, `.venv`. Describe them in
   data/README.md instead.
7. Pull `main` before starting work; resolve conflicts yourself; no force-push.
8. Do not make empty or filler commits. Commit history should reflect real
   contributions; the coordinator checks `git shortlog -sn` weekly.
9. Keep an `## Weekly log` entry in your area's notes (docs/logs/mN.md) - one
   short paragraph per week (done / blocked / next).
