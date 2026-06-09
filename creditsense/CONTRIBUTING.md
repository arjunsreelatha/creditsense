# Git Team Workflow Guide

> You've already created the 3 branches — great start!
> Here's everything you need to work as a team without breaking each other's code.

---

## The Golden Rule

> **main = safe, working code. Always.**
> Nobody pushes directly to main. Ever.

---

## Daily Workflow

### Starting Work
```bash
# Always sync your branch with latest main before coding
git checkout feature/your-branch
git pull origin main          # get latest safe code into your branch
```

### While Working
```bash
git add .                              # stage your changes
git commit -m "what you did"           # save a checkpoint
git push origin feature/your-branch   # back it up to GitHub
```

> Commit **often** — after every small working change. Not just at end of day.

### Merging Back to Main
Never merge yourself. Use a **Pull Request (PR)** on GitHub:
1. Push your branch
2. Go to GitHub → click **"Compare & pull request"**
3. Another teammate reviews it
4. Only then merge to main

---

## The Do's ✅

| Do | Why |
|---|---|
| Pull from main daily | Avoid falling behind and getting big conflicts |
| Write clear commit messages | `"fix login bug"` not `"fixed stuff"` |
| Push your branch daily | Backup + teammates can see your progress |
| Review each other's PRs | Catch bugs early, learn from each other |
| Talk before touching shared files | If two people edit same file → conflict |

---

## The Don'ts ❌

| Don't | Why |
|---|---|
| Push directly to main | Could break everyone's work |
| Work for 3 days without committing | Harder to undo mistakes |
| Ignore merge conflicts | They won't go away, only get worse |
| Delete branches before merging | You'll lose your work |
| Commit passwords/API keys | They stay in git history forever |

---

## Handling Merge Conflicts (Will Happen, Don't Panic)

When two people edit the same file, Git shows this:
```
<<<<<<< HEAD
your code here
=======
teammate's code here
>>>>>>> feature/their-branch
```

Just pick which version is correct (or combine both), delete the `<<<<`, `====`, `>>>>` lines, then:

```bash
git add .
git commit -m "resolved merge conflict"
```

---

## Recommended Team Rhythm

```
Monday morning      → everyone pulls latest main
During week         → commit + push daily to your own branch
Feature done        → open a Pull Request, teammate reviews
Friday / milestone  → one person merges approved PRs to main
```

---

## Quick Reference Card

```bash
git status                # what changed?
git log --oneline         # recent commits
git diff                  # see exact changes
git stash                 # temporarily save uncommitted work
git stash pop             # bring it back
```

---

> **Most important habit:** Pull before you code, push before you stop.
> That alone prevents 80% of team problems.