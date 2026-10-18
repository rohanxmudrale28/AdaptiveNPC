# Adaptive NPC: Player Behaviour Prediction and Counter-Strategy Generation

**Project report**

RL course mini project (B.Tech, Semester 7)

[NAMES / ROLL NOS / GUIDE / INSTITUTION]

[TODO: need number] (submission date)

---

## Contents

1. Introduction
2. Background
3. The game
4. System design
5. Implementation
6. Experiments and results *(stage B)*
7. Human evaluation *(stage B)*
8. Discussion, limitations and future work *(stage B)*
9. Conclusion *(stage B)*
10. Appendix *(stage B)*

**Conventions.** A win rate is measured from the NPC's side unless we say otherwise, and a drawn round counts as
half a win. "± x" is a 95% confidence interval.

---

## 1. Introduction

### 1.1 The problem: NPCs that players can read

Most enemies in games (non-player characters, or NPCs) follow behaviour a designer wrote by hand: always charge when
the player comes close, always block after attacking. This is easy to build and test, but players soon learn the
pattern and exploit it. The enemy never notices, because it has no idea what the player is doing.

We wanted to see whether an NPC can do better by paying attention to the player: learn their habits, guess their
next move, and pick a move that beats it. If the player changes style halfway through, the NPC should notice and
change too.

### 1.2 Our goal

We set out to build an NPC that (1) learns a player's habits, (2) predicts the player's next action, (3) picks a
counter-strategy from that prediction, and (4) adapts when the player switches style during a match.

The design is based on *Model-Based Opponent Modeling* (MBOM) by Yu et al. (NeurIPS 2022), a method for agents
that play against opponents who may themselves be learning or reasoning. We did not reproduce the paper's
experiments. We built a smaller version of its main ideas inside our own game, so that we could test them against
many kinds of opponents and let people play against the result.

### 1.3 What we built

- **A small 1v1 fighting game** on a 9×9 grid where two fighters move, attack, block and dodge. The rules run without
  graphics, so training is fast; a separate retro-style front end lets a person play with the keyboard.
- **Opponents to test against:** hand-written player styles ("archetypes" such as a rusher or a camper), players that
  learn during a match, offline-trained "exploiters" built to beat our NPC, and a lookahead opponent.
- **A player predictor:** a small recurrent network (GRU) that reads the last 8 steps and gives a probability for each
  of the player's 8 possible next actions. It keeps learning during a match.
- **Four NPC versions:** a hand-written NPC, plain PPO, PPO that also sees the predictor's output, and a scaled-down
  MBOM with a learned game model, "imagined" opponents that try to out-think the NPC, and a Bayesian mixer that
  decides which imagined opponent to believe.
- **An evaluation setup** with fixed seeds, where the pass/fail rules for the main MBOM questions were written down
  before running anything.

### 1.4 What we found, in one paragraph

Predicting the player worked well against the scripted styles, and the prediction-conditioned PPO NPC beat the
hand-written NPC. The full MBOM, however, did **not** beat its simpler baselines. Its imagination step did not help
against any opponent we tried, and in some groups it made the NPC slightly worse. MBOM with imagination switched
off did slightly better than the prediction-conditioned PPO NPC in some groups, and that small gain comes from the
settings MBOM uses to retrain its player predictor during a match (a larger learning rate, applied every 50 steps
instead of every 25), not from imagination. The Bayesian mixer did help compared with mixing the imagined opponents
equally. Chapter 6 gives the numbers and Chapter 8 discusses why. We have not yet tested the NPC on real people: the
human study is designed and its code is ready, but its results are still to come (Chapter 7).

---

## 2. Background

### 2.1 Reinforcement learning in a few words

In **reinforcement learning (RL)** an *agent* learns by trial and error. At each step it sees the situation (the
*observation*), picks an *action*, and receives a number (the *reward*) saying how good the result was. Its rule for
choosing actions is its *policy*, and training adjusts the policy to collect as much reward as possible over an
*episode* (for us, one round). A **value function** estimates how much reward the agent can still expect from a
given state: a "how well am I doing right now" score.

We train our NPCs with **Proximal Policy Optimization (PPO)**, from *Proximal Policy Optimization Algorithms* by
Schulman et al. (2017). PPO collects some experience with the current policy, improves the policy with a few small
steps, and limits how far each update can move it, so training stays stable. We use the Stable-Baselines3 PPO for
offline training, and a small PPO of our own for players that learn during a match.

### 2.2 Opponent modelling

In a two-player game the best move depends on what the other player will do. **Opponent modelling** means building
a model of the other player, usually a prediction of their next action or of their style, and using it to choose
better moves. *Autonomous Agents Modelling Other Agents: A Survey* by Albrecht and Stone (2018) covers the area. Our
NPC versions sit at three levels:

1. **No model.** The agent reacts to the current state only. Our plain PPO NPC works like this.
2. **Predict the opponent.** The agent predicts the opponent's next action from their past behaviour and uses the
   prediction in its own decision. *Opponent Modeling in Deep Reinforcement Learning* (DRON) by He et al. (2016) is an
   early example of giving a deep RL agent a representation of its opponent. Our prediction-conditioned PPO NPC
   follows the same general idea.
3. **Model an opponent who also reasons.** A clever opponent may think "the NPC expects me to attack, so I will
   block". Handling that means reasoning about the opponent's reasoning, possibly several levels deep. MBOM is built
   for this case.

### 2.3 The MBOM paper

*Model-Based Opponent Modeling* by Yu et al. (NeurIPS 2022) starts from the fact that an agent cannot know in
advance how clever its opponent is. So it keeps several guesses about the opponent at once, at different levels of
reasoning, and keeps checking which guess matches what the opponent actually does. It has four parts:

1. **An environment model:** a learned network that predicts the next state and reward from the current state and
   both players' actions, so the agent can imagine short futures without touching the real game.
2. **A level-0 opponent model:** a network trained on the opponent's real past actions to predict their next action.
   The paper calls the opponent models *imagined opponent policies* (IOPs).
3. **Recursive imagination:** to build level 1, MBOM asks "if the opponent knew the agent reacts to the level-0
   prediction, what would the opponent's best move be?". It tries each opponent action in the environment model,
   rolls a few steps forward with random continuations, and scores the outcome from the opponent's side (in a
   zero-sum game, the opponent's value is minus the agent's value). The best-scoring actions become training targets
   for level 1. Level 2 is built on top of level 1 the same way.
4. **Bayesian mixing:** after every real opponent action, MBOM checks how likely that action was under each level and
   updates a belief over the levels with Bayes' rule. A decayed average of these beliefs is turned into mixing weights
   with a "softer" softmax, and the agent acts on the weighted mix of the levels' predictions.

The agent's own policy is trained beforehand with PPO and **frozen** during play; only the opponent models and the
mixing weights change. The paper also describes ablations (versions with one part removed): no imagined levels,
levels fine-tuned on random actions, a single level, and uniform mixing. We built all of them. Our copy of the paper
has no appendices, so its hyperparameters were not available to us; all our settings are our own.

### 2.4 Where our project fits

Opponent-modelling methods are usually tested on research benchmarks. We put one inside a small playable game and
tested it against a wide mix of opponents: fixed styles, a style switcher, players that learn during a match,
opponents trained to beat our NPC, and (planned) real people.

---

## 3. The game

### 3.1 Arena and rules

Two fighters duel on a **9×9 grid**, each with **5 HP** and **6 stamina**, starting each round on a random tile in
their own half. Both choose an action at the same time and the game resolves them together. There are **8 actions**:

| action | what it does |
|---|---|
| idle | do nothing |
| up, down, left, right | move one tile and face that way (you turn even if the tile is blocked) |
| attack | hit the tile in front for 1 damage; no attacking again for 2 steps (cooldown) |
| block | guard the front only; costs 1 stamina per step held |
| dodge | dash up to 2 tiles in the facing direction, untouchable for that step; costs 3 stamina |

- **Parry:** attacking a blocking fighter from the front does no damage and **stuns the attacker for 2 steps**
  (a stunned fighter can only stand still).
- **Stamina** returns at 1 point every 2 steps. An action you can't afford, or an attack during cooldown, becomes
  idle.
- **Order within a step:** dodges, then moves, then blocks, then attacks. Two fighters moving into the same tile both
  stay put.
- **End of a round:** 0 HP, or **150 steps**; at the time limit the fighter with more HP wins, equal HP is a draw.
- **Match:** **30 rounds** by default. Learning NPCs and opponents keep what they learned across rounds of a match.

**Reward.** The NPC gets +1 for a win, −1 for a loss, 0 for a draw, plus +0.1 per damage dealt and −0.1 per damage
taken to make learning easier. The game is zero-sum: the opponent's reward is the negative of the NPC's.

**Observation.** The NPC sees 21 numbers: its position, the opponent's relative position, both facings, both HP,
stamina, cooldown and stun values, and how far into the round it is. The player's observation is the exact mirror,
so any style can play on either side.

All values live in `configs/env.yaml`; the rules are one 170-line file (`adaptive_npc/env/core.py`) that uses only
Python and NumPy.

### 3.2 The rock-paper-scissors loop

The rules make sure no action is always best:

- **Attack beats moving:** stepping in front of an opponent gets you hit.
- **Block beats attack:** a frontal attack into a block does nothing and stuns the attacker.
- **Flanking or waiting beats block:** a block only covers the front, and holding it drains stamina.

Dodge sits outside the triangle: it escapes any attack but costs half the stamina bar. Because every action has a
counter, knowing the opponent's next move is genuinely useful, which is what a predicting NPC needs.

### 3.3 The retro game shell

The playable front end (Pygame) only reads the game state and draws it. Everything is drawn on a 320×240 surface
in a fixed 16-colour palette and scaled three times to a 960×720 window. It has a title screen, an opponent-select
screen, a HUD with HP, stamina and score, pause and results screens, synthesized sound effects and two chiptune
tracks (generated in code, no audio files), and a **demo panel** (Tab). The panel shows live bars for the NPC's
prediction of the player's next action, highlights the action actually taken, keeps a running accuracy, and against
the MBOM enemy shows the mixer weights over the opponent-model levels with the 0.79 cap marked (Section 4.7).
Controls: WASD or arrows to move, J attack, K block, L dodge.

**Tick rate.** Human play runs at **4 game steps per second** by default, with the screen redrawn at 60 fps. We
lowered it from 8 after a first play-test found 8 too hard. The tick rate only affects human play; training and
evaluation run as fast as the computer allows.

[SCREENSHOT: title screen]

[SCREENSHOT: opponent-select screen with the five named enemies and their tags]

[SCREENSHOT: a fight in progress with the HUD]

[SCREENSHOT: demo panel open against ECHO, showing predicted-action bars, running accuracy and the α weights with the cap line]

### 3.4 The named enemies

| enemy | what it really is | tag |
|---|---|---|
| TRAINING DUMMY | the camper archetype | TUTORIAL |
| RUSTY | the hand-written scripted NPC | EASY |
| REFLEX | plain PPO (`ppo_none`) | MEDIUM |
| SEER | prediction-conditioned PPO (`ppo_predicted`) | HARD |
| ECHO | MBOM | HARD |

The tags order the enemies by sophistication; they were not measured against people. ECHO shares SEER's tag on
purpose, because our results do not show MBOM to be stronger than `ppo_predicted`.

A **blind mode** (`--blind`) hides names and the panel, gives every enemy the same look and sound, and logs which
NPC was really played, for the human study.

**Caveat about ECHO.** An MBOM update takes about 3 seconds. To keep the game smooth, the playable ECHO runs it in a
background thread, so its result arrives a few steps late, and an update is skipped if the previous one is still
running. The playable ECHO is therefore only an **approximation** of the MBOM we evaluated. All MBOM numbers in this
report come from the synchronous version, which the game can also run (`ui.mbom_async=false`), with short pauses.

### 3.5 Game balance

Before training anything we checked that no scripted style wins everything; otherwise "find the counter-strategy"
would have one boring answer.

**Method.** A round-robin between 10 players: the 6 training and 2 held-out archetypes (Section 4.1), the scripted
NPC and a random player. Each pair played **200 rounds per seed on each side, with seeds 0, 1 and 2**. We fixed four
checks in `configs/balance.yaml` before tuning:

1. **No side bias:** in mirror matches the first slot wins 45–55%.
2. **No dominant style:** no style wins at least 65% against *every* other, on either the draw-adjusted or the
   decisive win rate (wins ÷ (wins + losses)).
3. **Every non-random style wins at least one matchup and loses at least one.**
4. **Every non-random style has a counter** that beats it at least 60%, on both win-rate measures.

**Tuning.** The first run failed: the counter-puncher won everything (1.000 against the rusher, 0.999 against the
kiter), and its best counter, the scripted NPC, reached only 0.323. A diagnosis showed the rules already allow a
counter, since a side attack gets past a block, so we changed the players and left the rules alone. We added the
**flanker** (attacks from the side) as a training archetype; on its own it only drew level with the counter-puncher
(0.494). We then made the counter-puncher react one step late with 20% random actions. After that, all four checks
passed. Each iteration is logged in `results/balance/log.md`.

![Figure 3.1: Round-robin win-rate matrix](figures/balance_winrate_heatmap.png)

*Figure 3.1. Win rate of the row player against the column player. 200 rounds per pair per seed, seeds 0–2, both
sides, draws = ½. Source: `results/balance/winrate_matrix.csv`.*

**Result (Figure 3.1).** Mirror matches ranged from 0.4825 to 0.5058 in the first slot. No style is dominant, every
non-random style wins and loses at least once, and each has a counter: for example the counter-puncher beats the
rusher 0.90 and the kiter 0.96, the scripted NPC beats the counter-puncher 0.65, and the camper beats the scripted NPC
0.64. The scripted NPC is a reasonable baseline: it beats the random player 0.999 but loses to the camper (0.365).

The balance is not perfect:

- **The flanker is weak overall.** It beats the counter-puncher (0.636), repeater and mixer but loses to most others,
  e.g. 0.010 against the rusher and 0.020 against the kiter.
- **Passive styles mostly draw** against each other (camper, repeater, mixer, random; Figure 3.2). 20 pairs,
  including mirror pairs, draw more than half their rounds, so their win rates sit near 0.5.
- **The counter-puncher still wins 90% or more against predictable styles** (0.902 against the rusher, 0.961 against
  the kiter, 0.994 against the repeater, 0.982 against the mixer). It is not dominant, because the flanker and the
  scripted NPC beat it, but lowering these numbers would need a change to its logic.

![Figure 3.2: Draw rate per pair](figures/balance_draw_rate_heatmap.png)

*Figure 3.2. Share of drawn rounds per pair in the same round-robin. Source: `results/balance/draw_rate_matrix.csv`.*

---

## 4. System design

Figure 4.1 shows how the parts fit together. The game core knows nothing about who is playing: every NPC and every
opponent plugs into one of its two sides.

```
                     +---------------------------------------------+
                     |  Game core (rules only, no graphics)         |
                     |  9x9 grid, 8 actions, HP, stamina, parry     |
                     +-------------+-----------------------+-------+
                                   |                       |
          NPC side                 v                       v        opponent side
 +-------------------------------------------+   +--------------------------------+
 | scripted NPC     hand-written rules        |   | 8 scripted archetypes          |
 | ppo_none         PPO on the observation    |   | naive / reasoning learners     |
 | ppo_predicted    PPO + predictor output    |   | exploiters (frozen)            |
 | MBOM             frozen ppo_predicted      |   | lookahead opponent             |
 |                  + mixed opponent models   |   | human (keyboard)               |
 +---------------------+---------------------+   +--------------------------------+
                       |
   MBOM during a match v
 +-------------------------------------------------------------------------------+
 | real opponent actions --> phi0 (GRU predictor, fine-tuned on real actions)    |
 | env model Gamma + imagination:                                                |
 |    opponent best response to NPC-using-phi0 --> trains phi1                   |
 |    opponent best response to NPC-using-phi1 --> trains phi2                   |
 | phi0, phi1, phi2 --> Bayesian mixer (weights alpha) --> mixed prediction      |
 |                  --> frozen PPO policy --> NPC action                         |
 +-------------------------------------------------------------------------------+
 Pygame shell, demo panel and sound only read the game state.
```

*Figure 4.1. Block diagram of the system.*

### 4.1 Opponents

**Scripted archetypes.** Short hand-written rules, most with 5% random actions so they are not perfectly
predictable.

- *Training archetypes* (used to train the predictor and the PPO NPCs): **rusher** (charges and attacks whenever it
  faces the opponent); **camper** (stays put, blocks when the opponent is within 2 tiles, pokes 30% of the time);
  **repeater** (a fixed 8-step cycle: toward, toward, attack, idle, block, block, away, dodge); **mixer** (random
  actions from fixed weights, 25% attack and 20% block); **flanker** (moves to the opponent's side and attacks
  there); **switcher** (changes style every 10 rounds: rusher, camper, repeater, mixer).
- *Held-out archetypes* (never trained on, used to test generalisation): **kiter** (hits, then backs off while its
  attack recharges) and **counter-puncher** (blocks at close range and punishes after the opponent attacks; after
  balancing it reacts one step late with 20% random actions).

**Scripted NPC.** The hand-written baseline: close in, attack when ready, block while the attack recharges if the
opponent faces it. No randomness. This is the kind of NPC the project tries to improve on.

**Learning opponents.** To see how the NPC copes with players who learn, we built two:

- The **naive learner** is a PPO policy **pretrained against the scripted NPC** (1M steps, seeds 0–2) that keeps
  training during a match with our small PPO, updating every 64 steps at learning rate 3×10⁻⁴.
- The **reasoning learner** also has its own GRU model of the NPC and feeds that model's prediction into its policy.
  Both parts keep learning during the match.

We picked the online learning rate and update interval with a sweep (3×10⁻⁴, 10⁻³ or 3×10⁻³; every 32 or 64 steps),
choosing the smallest rate that changed the policy, still beat a random player, and did not collapse. Two points
matter later: their pretraining against the scripted NPC gives them a head start against that NPC, and a 30-round
match gives them only about 600–1,000 steps to adapt.

**Exploiters.** To test whether our best NPC can be beaten by an opponent that knows its weaknesses, we trained
three exploiters (seeds 0, 1, 2). Each starts from the pretrained naive learner and trains offline with PPO for **1
million steps** against a frozen `ppo_predicted` NPC, the checkpoint with seed 0. In evaluation they are frozen and
sample their actions. `exploiter_seed1` is the **development** exploiter, used only for tuning and go/no-go checks;
`exploiter_seed0` and `exploiter_seed2` are used for the **final claims**. Because all three were trained against NPC
seed 0, results at NPC seed 0 and at NPC seeds 1–4 are always reported separately.

**Lookahead opponent.** For one test (H2, Chapter 6) we built an opponent that reasons exactly the way MBOM's level 1
imagines. Each step it assumes the NPC is plain `ppo_predicted`, predicts the NPC's move with that frozen policy and
a frozen copy of the predictor, tries each of its own 8 actions two steps ahead in the true rules (16 random
continuations each, as in MBOM), and plays the best one; 5% of its actions are random. It does not learn. Its settings
were fixed before the test and not changed afterwards.

### 4.2 The player predictor

The predictor "learns the player's habits". For each of the last **8 steps** it reads the NPC's 21-number
observation plus the previous NPC and player actions, and a single-layer **GRU** (64 units, a recurrent network that
keeps a short memory of what it has read) outputs a probability for each of the player's 8 next actions.

It is trained offline on play between the training archetypes and two NPCs (the scripted NPC and a random NPC), 200
rounds per archetype. During a match it keeps learning: every 25 steps it takes 3 training steps on the last 256
examples, at learning rate 3×10⁻⁴. We chose these online settings with a sweep over seven settings
(`results/predictor/main/online_sweep.csv`). This is `ppo_predicted`'s online setting.

### 4.3 The NPC versions

| name | decides from |
|---|---|
| `scripted` | the hand-written rules above |
| `ppo_none` | PPO on the 21-number observation |
| `ppo_predicted` | PPO on the observation plus the predictor's 8 probabilities |
| `mbom` | the frozen `ppo_predicted` policy, fed a mix of three opponent models instead of the predictor alone |

A debug version, `ppo_oracle`, is given the player's *true* next action. It shows the ceiling for a
prediction-based NPC and is never presented as a real method.

Each PPO NPC is trained with Stable-Baselines3 for **2 million steps**, 8 games in parallel, against the training
archetypes (a new one drawn each round), with two 64-unit layers for policy and value. We trained **5 seeds** (0–4)
of each. The predictor is frozen during training. In evaluation the PPO NPCs act greedily, and in the MBOM
experiments `ppo_predicted` fine-tunes its predictor online (Section 4.2) so that it is a fair baseline for MBOM.

### 4.4 The environment model Γ

MBOM needs a model of the game to imagine futures in. Our learned model **Γ** (Gamma) takes the full game state (15
whole numbers: positions, facings, HP, stamina, cooldowns, stun timers and the step counter) plus both actions, and
predicts the next state and the reward. It is a network with two 256-unit hidden layers.

Because every part of the state is a small whole number, Γ treats each one as a classification ("does HP change by
−1, 0 or +1?"), and snaps its prediction to a legal state before feeding it back in, so errors don't drift into
impossible states. It learns the small damage part of the reward; the ±1 win/loss part is computed from the predicted
next state. Our first version predicted the reward as one number and got 0% of the end-of-round rewards right, which
is why we split it.

Γ was trained for 40 epochs on recorded play between many NPCs (scripted, random, and the PPO versions over 5 seeds)
and many opponents (training archetypes, random, the learners, exploiters 0 and 1): 86.7k training transitions, 34.3k
validation transitions with different seeds, and 23.2k against opponents Γ never saw (kiter, counter-puncher,
exploiter 2). We added 57.7k transitions where both sides act at random, 40% of the training mix, so Γ also knows
unusual situations. A debug switch replaces Γ with the true rules; we used it once, to check whether Γ's errors
explain MBOM's results, and never for reported results.

### 4.5 MBOM in our game

MBOM's policy is the frozen `ppo_predicted` policy. Instead of the predictor's output, it receives a weighted mix of
three opponent models (levels). Only the levels and the weights change during a match.

**Levels.** **φ0** is the pretrained predictor, fine-tuned on the opponent's real actions every **50 real steps** (3
training steps on the last 256 examples, learning rate **10⁻³**). This differs from `ppo_predicted`'s online setting
(every 25 steps at 3×10⁻⁴), and the difference turns out to matter (Chapter 6). **φ1** and **φ2** start each match as
copies of φ0. At each update, φ1 is trained to predict the opponent's best response to an NPC that trusts φ0, and φ2
the best response to an NPC that trusts φ1.

**Imagination.** For each of the last 256 real states and each of the opponent's 8 actions: the opponent plays that
action and the NPC answers with its frozen policy conditioned on the level below; Γ then rolls **2 more steps**
(k = 2) with random opponent moves, **16 times**; each rollout is scored from the opponent's side (its rewards plus
minus the NPC's value estimate at the end); and an action's score is its best rollout. The 8 scores become soft
targets through a sharp softmax (τ = 0.1). All 8 actions share the same random rollouts, so they are compared fairly.

### 4.6 The Bayesian mixer

After each real opponent action the mixer (1) checks how likely that action was under each level, (2) updates a
belief over the levels with Bayes' rule, using a moving average of past beliefs as the prior (weight 0.9, with a floor
of 0.01 so a long-wrong level can come back), (3) keeps a decayed average Ψ of the beliefs (decay 0.9), and (4) sets
the weights α = softmax(Ψ / T) with **T = 0.5**. The policy receives α₀·φ0 + α₁·φ1 + α₂·φ2.

### 4.7 The α cap

Ψ is an average of probability distributions, so its values add up to 1. With T = 0.5 the largest weight any level
can get is e² / (e² + 2) ≈ **0.787**, even when the mixer is certain. So the mixer can never fully commit: the other two
levels always keep about 21% between them. Chapter 6 shows this cap matters.

On synthetic data with a known answer (20 seeds), the mixer worked as designed: the true level took the lead for good
by step 17, and after a switch the new level led after a median of 50 steps when the levels overlapped strongly
(20 steps when they overlapped less). Most of the lag comes from the slowly moving prior.

Feeding the frozen policy a blend of predictions is something it never saw in training. Before the main runs we
checked, with a rule fixed in advance, that blended inputs don't break it against the training archetypes. They did
not, so we kept the mixed input (details in Chapter 6).

### 4.8 Variants

| variant | what changes |
|---|---|
| `mbom_no_imagination` | only φ0, still fine-tuned on MBOM's schedule. Against `mbom` it shows what imagination adds; against `ppo_predicted` it shows what MBOM's φ0 schedule adds |
| `mbom_uniform_mix` | equal weights on all levels instead of the Bayesian mixer |
| `mbom_single_level1`, `mbom_single_level2` | all weight on one imagined level |
| `mbom_random_finetune` | levels fine-tuned on random actions instead of best responses |
| `mbom_top_level` | the policy gets the highest-weighted level alone instead of the mix |

The mixer and imagination settings were tuned only on the development set (training archetypes and
`exploiter_seed1`).

---

## 5. Implementation

### 5.1 Stack and rules

Everything is Python and runs **locally on a CPU**, with **no external APIs or online services**: PyTorch
(predictor, Γ, MBOM), Gymnasium (environment interface), Stable-Baselines3 (offline PPO), Pygame (game and sound),
NumPy, SciPy (95% t-intervals), pandas and Matplotlib, PyYAML and pytest. We tested with Python 3.12, torch 2.14,
numpy 2.5, gymnasium 1.4, pygame 2.6, scipy 1.18 and pytest 9.1. Stable-Baselines3's save format did not load with
this torch version, so PPO checkpoints are stored as plain PyTorch `.pt` files. The machine was a desktop with an
Intel i5-14400F (6 performance and 4 efficiency cores).

We followed a few rules throughout:

- **Rules apart from drawing.** The game core never imports Pygame, so training runs headless and fast.
- **No settings in code.** Every seed and hyperparameter is in a YAML file in `configs/`.
- **Reproducible runs.** Fixed seeds; every run saves its results to `results/` with a copy of its config and the
  git commit hash.
- **Small steps.** Milestones M0–M14 (`PLAN.md`), each built, smoke-tested with a one-minute config, and committed.
- **Pre-registration.** For the MBOM tests the pass/fail rule was written into `PLAN.md` and the config before any run.

### 5.2 Speed

Random players on both sides (`results/benchmarks/env_speed.csv`):

| setup | steps per second |
|---|---|
| game core alone | 399,288 |
| one Gymnasium environment | 121,940 |
| 8 environments in one process | 80,703 |
| 8 environments in separate processes | 27,731 |

This is far above our 10,000 steps/s target. Separate processes were slower because a game step is so cheap that
passing data between processes costs more, so we kept the single-process setup.

MBOM is much more expensive. One update takes about 3.0 s with 10 threads (6.3 s with one), about 60% of it in the
opponent-model GRUs and about 27% in Γ. A 30-round MBOM match took 23–56 s (mean 38 s) with 10 threads, against
1.5–2.7 s for `ppo_predicted`. The final experiment grid took about 49 process-hours, run as parallel single-thread
processes.

### 5.3 Code layout

```
adaptive_npc/   the package
  env/          game core (no Pygame), observations, Gymnasium wrapper, vectorised envs, match runner
  players/      scripted archetypes, learners, lookahead opponent
  models/       GRU predictor, environment model, small online PPO
  npc/          PPO NPC (3 input modes), scripted NPC, live wrappers for the game; mbom/ (levels,
                imagination, mixer, agent)
  eval/         metrics, experiment runner, balance, predictor and env-model evaluation
  render/, ui/, audio/   Pygame drawing, game shell and human-study session, synthesized sound
configs/        every seed and hyperparameter (YAML); experiments/ holds the smoke configs
scripts/        command-line entry points (python -m scripts.<name> --config ...)
tests/          pytest suite, headless
models/final/   final trained checkpoints
results/        saved metrics, config snapshots and plots per experiment
report/         this report and its figures
```

### 5.4 Testing

The suite has **156 tests** and runs headless in about a minute. It checks every game rule, that a seed always
gives the same game, that each archetype behaves differently and the switcher switches on time, that the scripted NPC
beats a random player more than 80% of the time, that Γ's rollouts match the true rules, that the mixer finds the
right level, follows a switch and respects the α cap, that imagination finds the known best response in two
hand-built states (checked by brute force with the real rules), and that the game shell, panel, sound and
human-study logging work.

### 5.5 How to run it

From the repository root in PowerShell (Appendix A has the full list):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# copy the final models from models\final\ into models\ (README, "Model checkpoints")
python -m pytest -q                                  # tests
python -m scripts.play                               # play: title screen, then pick an enemy
python -m scripts.play --opponent mbom --panel on    # play ECHO with the demo panel open
```

Every training and evaluation script has a smoke config in `configs/experiments/` that runs end to end in about a
minute.

---

*End of stage A. Chapters 6–10 follow in stage B.*

<!-- Stage B: chapters 6-10 -->

---

## Appendix C (draft, stage A numbers only). Numbers used and their sources

Completed in stage B with the numbers from Chapters 6–10.

| number | meaning | source |
|---|---|---|
| 9×9; 5 HP; 6 stamina; +1 stamina per 2 steps | arena and resources | `configs/env.yaml` |
| 1 damage, cooldown 2; block 1 stamina/step; parry stun 2; dodge 3 stamina, 2 tiles | actions | `configs/env.yaml` |
| 150 steps per round; 30 rounds per match | round and match length | `configs/env.yaml`; `configs/mbom.yaml`; PLAN.md Decision 5 |
| ±1, 0, ±0.1 per damage | reward | `configs/env.yaml` |
| 21 | observation size | `adaptive_npc/env/observations.py` (`OBS_DIM`) |
| 170 lines | rules file | `adaptive_npc/env/core.py` |
| 4 steps/s (from 8); 60 fps | human-play tick rate, render rate | `configs/render.yaml`; PLAN.md Decision 20 |
| 320×240, ×3, 960×720; 16 colours | retro shell | `configs/ui.yaml` |
| about 3 s | MBOM update time in the game | PLAN.md Known limitations ("~3 s") |
| 200 rounds, seeds 0–2; 0.45–0.55; 0.65; 0.60 | balance design | `configs/balance.yaml` |
| 1.000; 0.999; 0.323 | first balance run | `results/balance/log.md` (iteration 0) |
| 0.494 | flanker vs counter-puncher, flanker only | `results/balance/log.md` (iteration 1) |
| 0.4825–0.5058 | mirror matches, first slot | `results/balance/slot0_matrix.csv` (diagonal) |
| 0.90; 0.96; 0.65; 0.64 | counter examples | `results/balance/report.txt` |
| 0.999; 0.365 | scripted NPC vs random, vs camper | `results/balance/winrate_matrix.csv` |
| 0.636; 0.010; 0.020 | flanker vs counter-puncher, rusher, kiter | `results/balance/winrate_matrix.csv` |
| 20 pairs | pairs above 50% draws | `results/balance/report.txt` |
| 0.902; 0.961; 0.994; 0.982 | counter-puncher vs rusher, kiter, repeater, mixer | `results/balance/winrate_matrix.csv` |
| 5%; 2 tiles; 30%; 8-step pattern; 25% / 20%; 10 rounds; delay 1, 20% | archetype settings | `configs/players.yaml` |
| 1M steps, seeds 0–2; 3×10⁻⁴, every 64; sweep values | learners | `configs/learners_pretrain.yaml`; `configs/learners.yaml` |
| 600–1,000 steps | learner experience per match | PLAN.md Known limitations |
| 1M steps; seeds 0–2; NPC seed 0 | exploiters | `configs/exploiter.yaml` |
| k = 2; 16; 5% | lookahead opponent | `configs/mbom.yaml` (`lookahead:`) |
| 8 steps; 64 units; 200 rounds; every 25 steps, 3 steps, 256, 3×10⁻⁴; 7 settings | predictor | `configs/predictor.yaml` |
| 2M steps; 8 envs; 64-unit layers; 5 seeds | PPO NPC training | `configs/npc_ppo.yaml` |
| 15 numbers; 256-unit layers; 40 epochs | Γ | `adaptive_npc/models/env_model.py`; `configs/env_model.yaml` |
| 0% | first Γ version's end-of-round reward accuracy | PLAN.md M7 |
| 86.7k; 34.3k; 23.2k; 57.7k; 40% | Γ data | PLAN.md M7; `configs/env_model.yaml` |
| 3 levels; every 50 steps, 3 steps, 256, 10⁻³; k = 2; 16; τ = 0.1 | MBOM | `configs/mbom.yaml` |
| 0.9; 0.01; 0.9; T = 0.5; ≈ 0.787 | mixer | `configs/mbom.yaml`; PLAN.md M8 |
| 20 seeds; step 17; median 50 / 20 steps | mixer on synthetic data | PLAN.md M8 |
| 399,288; 121,940; 80,703; 27,731 | steps per second | `results/benchmarks/env_speed.csv` |
| 10,000 steps/s | speed target | PLAN.md M1 |
| 3.0 s / 6.3 s; ≈ 60%; ≈ 27%; 23–56 s (mean 38); 1.5–2.7 s | MBOM speed | PLAN.md M8 benchmark |
| ≈ 49 process-hours | final grid | README.md; PLAN.md M10 |
| 156 tests, about 1 minute | tests | README.md |
| Python 3.12, torch 2.14, numpy 2.5, gymnasium 1.4, pygame 2.6, scipy 1.18, pytest 9.1 | versions | README.md |
| i5-14400F, 6 + 4 cores | machine | PLAN.md M10 |
