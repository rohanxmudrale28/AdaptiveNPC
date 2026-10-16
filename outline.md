# Adaptive NPC: Player Behaviour Prediction and Counter-Strategy Generation

## Problem
Game NPCs use predefined behaviours, so players learn to exploit them. Goal: an NPC that learns
a player's habits, predicts their next action, and generates counter-strategies, adapting when
the player changes style.

## Base paper
references/MBOM_NeurIPS2022.pdf. We reimplement its core ideas in our own game, not a
benchmark: recursive imagination, Bayesian mixing of opponent models, and evaluation against
fixed, naive-learner and reasoning-learner opponents.

## Objectives
1. Build a 1v1 arena game env with scriptable, learning and human players
2. Predict player actions from behaviour history
3. Generate counter-strategies with RL
4. Adapt when the player changes style mid-match
5. Evaluate against baselines and ablations, then with real humans

## Game
- 1v1 2D arena in Pygame, discrete actions: move (4 dirs), attack, block, dodge
- Cooldowns/stamina so actions have trade-offs
- Gymnasium-style reset()/step(); rendering separate so training runs headless and fast
- Keyboard-playable by a human against the NPC
- Live demo panel: predicted next action and belief weights over player types

## Players (opponents)
- Fixed-policy scripted archetypes: aggressive rusher, camper, fixed-pattern repeater,
  random mixer, style switcher (changes behaviour mid-match)
- Naive learner: player that updates its policy with PPO during play
- Reasoning learner: player that models the NPC and conditions its policy on that prediction
- Held-out archetypes used only for testing generalization

## Components
1. Player model: GRU predicting the player's next action from recent action/state history
2. NPC policy: PPO conditioned on the predicted action distribution
3. MBOM: env model (next state, reward), level-0 opponent model, recursive imagination via
   short rollouts (2-3 levels), Bayesian mixing of imagined opponent policies

## Literature survey (for report)
Opponent modeling (DRON, DPIQN, policy representations), Theory of Mind (ToMnet, SOM),
learning-aware methods (LOLA, Meta-PG, Meta-MAPG), MBOM, adaptive game AI.
Gap: these are evaluated on research benchmarks, we apply them to a playable game with
style-switching and learning players.

## Contribution
Scaled-down MBOM-style adaptive NPC in a custom game, with comparison of scripted NPC,
plain PPO, prediction-conditioned PPO, and full MBOM.

## Experiments
- Opponent types: fixed policy, naive learner, reasoning learner
- Methods: scripted NPC, plain PPO, prediction-conditioned PPO, MBOM, MBOM ablations
- Ablations: no recursive imagination, no Bayesian mixing, uniform mixing
- Unseen-opponent generalization (held-out archetypes)
- Adaptation curves: win rate and prediction error vs time after a style switch
- Human evaluation: repeated matches, NPC win rate and accuracy across matches
- 5 random seeds, mean and 95% confidence intervals

## Metrics
Prediction accuracy (and KL divergence), NPC win rate, adaptation time after style switch,
average episode return.

## Deliverables
Working playable game, code repo with README, trained models, result plots, report,
presentation and demo.

## Constraints
Local only, no external APIs. Windows + PowerShell. Python, PyTorch, Gymnasium, Pygame,
Stable-Baselines3.

## Open questions (propose defaults)
Exact action set, reward shaping, arena size, number of MBOM levels, network sizes.
