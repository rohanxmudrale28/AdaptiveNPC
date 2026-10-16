# Project
Adaptive NPC: an RL agent in a small 1v1 Pygame arena game that learns a player's habits,
predicts their next action, and generates counter-strategies, adapting when the player changes
style. Scaled-down reimplementation of ideas from MBOM (references/MBOM_NeurIPS2022.pdf).
Full spec in outline.md.

# Stack
Python, PyTorch, Gymnasium, Pygame, Stable-Baselines3, NumPy, Matplotlib. Everything local.
No external APIs.

# Rules
- Complete, ready-to-run files
- PowerShell-compatible commands (Windows)
- Small targeted edits, don't rewrite working code
- Ask before adding a new dependency
- Keep the game env separate from rendering so training runs headless and fast
- Seeds, configs and hyperparameters go in config files, never hardcoded in scripts
- Every experiment is reproducible: fixed seeds, results saved to results/
- Commit after each working step with a clear message

# Workflow
- Plan first, then build one piece at a time, test it, commit, move on
- I run long training myself. Don't watch or read long training logs. I'll paste the summary or error
- Prefer specific files over scanning the whole project
