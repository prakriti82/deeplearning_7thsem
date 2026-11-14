# How to commit (first time)

Coordinator (M10), once:
    cd p6-forecasting
    git init -b main
    git add .
    git commit -m "chore: initial project scaffold"
    git remote add origin https://github.com/<org-or-user>/<repo>.git
    git push -u origin main
Then add all members as collaborators on GitHub.

Every member:
    git clone https://github.com/<org-or-user>/<repo>.git && cd <repo>
    git checkout -b feature/mN-<topic>
    # work on YOUR files (see docs/TEAM_PLAN.md), then:
    git add <your files>
    git commit -m "<area>: what you changed"
    git push -u origin feature/mN-<topic>
    # open a Pull Request on GitHub; another member reviews and merges
    git checkout main && git pull        # before starting the next piece of work

Set your identity once so commits are credited to you:
    git config user.name "Your Name"
    git config user.email "your-github-email@example.com"
