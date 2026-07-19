# FADE — 3.1T Total / 44.5B Active Parameter Code Model

> **Status: architecture only, untrained.** This repo contains a real,
> working model architecture and training pipeline. It does **not**
> contain trained weights — running inference on it right now would
> produce random, meaningless output. See "What's real here vs. what you
> still need" below for exactly what's required to change that.

## Specs (all decided, since none were specified)


| Property | Value | Why |
|---|---|---|
| Total parameters | 3.098T | matches your target |
| Active parameters per token | 44.55B | matches your target |
| Context window | 16,384 tokens | shared budget for input + output combined — this is a decoder-only model, there's no separate "input limit" and "output limit," just one context window you split between prompt and generation |
| Max output tokens | not fixed — set at generation time via `max_new_tokens` in `model.py`, bounded by `16384 - your_prompt_length` | generation length is a runtime choice, not an architectural constant |
| Architecture | RMSNorm, RoPE, Grouped-Query Attention, MoE-SwiGLU (160 experts, top-2 routing) | current best-practice recipe for efficient large models |
| Training objective | causal LM + 50% fill-in-the-middle | FIM is what makes a code model good at mid-function autocomplete, not just appending text |

Run `python config.py` any time to re-verify these numbers — it's pure arithmetic, no GPU needed.

## What's real here vs. what you still need

**Real and correct:** the architecture (`model.py`), the training loop
(`train.py`), the FIM data transform (`data.py`). These are legitimate
implementations of the same techniques real MoE models use.

**Not something any code can hand you**, regardless of budget:

1. **Sharding infrastructure.** At 3.1T parameters, the model does not fit
   on one GPU, or realistically one node. Training this requires
   expert-parallelism and model-parallelism frameworks (DeepSpeed-MoE,
   Megatron-Core, or similar) to split experts and layers across hundreds
   to thousands of GPUs, with the communication patterns to match. That's
   a separate, substantial engineering layer on top of this code — normally
   a dedicated infrastructure team's job.

2. **Data.** A model this size needs a very large, high-quality, deduplicated
   code corpus (many languages, real repositories, permissively licensed or
   rights-cleared) — assembling and cleaning that is normally many months
   of dedicated work.

3. **Compute time.** Even with a large GPU fleet, training to convergence
   at this scale is realistically months, not days — this is true for every
   organization that has trained models at anywhere near this scale.

4. **Post-training.** Instruction-tuning and preference optimization so it
   actually behaves like a helpful coding assistant rather than a raw
   text-completion engine — another substantial phase after pretraining
   finishes.

## Honest framing

This is a genuine, correct architecture at a scale that, as far as public
information goes, nobody has built specifically for code. That's worth
being clear-eyed about in both directions: the code isn't fake or a toy,
but reaching a *working* Fade also isn't a code problem — it's a
data + infrastructure + time problem that scales with the parameter count
you choose, and at 3.1T that's a genuinely enormous undertaking, not a
weekend project regardless of budget.
