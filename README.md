<div align="center">

# 🌒 Fade

**A 3.1T-parameter Mixture-of-Experts architecture for code generation**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-architecture--only%2C%20untrained-orange)](#status)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](requirements.txt)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-red)](requirements.txt)

[Support this project ☕](#-support-this-project) · [About the founder](ABOUT.md) · [Contributing](CONTRIBUTING.md) · [Changelog](CHANGELOG.md)

</div>

---

> **Status: architecture only, untrained.** This repo contains a real,
> working model architecture and training pipeline. It does **not**
> contain trained weights — running inference on it right now would
> produce random, meaningless output. See [What's real vs. what's still needed](#whats-real-here-vs-what-you-still-need) below.

## Overview

Fade is an open architecture for a large-scale Mixture-of-Experts code
model: many experts exist in the network, but only a small subset activate
per token, keeping actual compute manageable relative to total capacity.
It uses the same family of building blocks as current state-of-the-art
open models (RMSNorm, RoPE, Grouped-Query Attention, SwiGLU experts),
combined with a fill-in-the-middle training objective specifically for code.

## Specs

| Property | Value | Why |
|---|---|---|
| Total parameters | 3.098T | many experts held in memory |
| Active parameters per token | 44.55B | only top-2 experts run per token |
| Context window | 16,384 tokens | shared budget for input + output — decoder-only models don't have separate input/output limits, just one pool you split between prompt and generation |
| Max output tokens | set at generation time via `max_new_tokens`, bounded by `16384 − prompt length` | generation length is a runtime choice, not fixed in the architecture |
| Architecture | RMSNorm · RoPE · Grouped-Query Attention · MoE-SwiGLU (160 experts, top-2 routing) | current best-practice recipe for efficient large models |
| Training objective | causal LM + 50% fill-in-the-middle | FIM is what makes a code model good at mid-function completion, not just appending text |

Run `python config.py` any time to re-verify these numbers — pure arithmetic, no GPU required.

## Quickstart

```bash
git clone https://github.com/YOUR-USERNAME/fade-model.git
cd fade-model
pip install -r requirements.txt
python config.py     # verify parameter counts
python model.py       # build the model and print its parameter count
```

## What's real here vs. what you still need

**Real and correct:** the architecture (`model.py`), the training loop
(`train.py`), and the FIM data transform (`data.py`). These are genuine
implementations of the techniques real MoE models use — not simplified
stand-ins.

**Not something any codebase can hand you**, regardless of budget:

1. **Sharding infrastructure** — at 3.1T parameters the model doesn't fit
   on one GPU or one node. Training needs expert-parallel and
   model-parallel frameworks (DeepSpeed-MoE, Megatron-Core, or similar)
   to split the model across hundreds-to-thousands of GPUs.
2. **Data** — a large, high-quality, deduplicated, rights-cleared code
   corpus across many languages.
3. **Compute time** — realistically months of large-scale training, not
   days, at this parameter count.
4. **Post-training** — instruction-tuning and preference optimization to
   turn a raw text-completion model into a genuinely helpful assistant.

## Roadmap

- [ ] Reference expert-parallel training config (DeepSpeed-MoE)
- [ ] Tokenizer training script for code corpora
- [ ] Smaller reference configs (1B / 10B) for testing on modest hardware
- [ ] Evaluation harness (HumanEval, MBPP, etc.)
- [ ] Post-training / instruction-tuning pipeline

Contributions toward any of these are very welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).

## ☕ Support this project

Building, documenting, and eventually training models at this scale takes
real time and (eventually) real compute cost. If this project is useful
to you, donations go toward continued development and — longer term —
actually training a model on this architecture:

<a href="https://www.buymeacoffee.com/YOUR-USERNAME"><img src="https://img.shields.io/badge/Buy%20me%20a%20coffee-support-ffdd00?logo=buy-me-a-coffee&logoColor=black" alt="Buy Me a Coffee"></a>
<a href="https://github.com/sponsors/YOUR-GITHUB-USERNAME"><img src="https://img.shields.io/badge/GitHub%20Sponsors-support-EA4AAA?logo=github-sponsors&logoColor=white" alt="GitHub Sponsors"></a>

*(Replace `YOUR-USERNAME` / `YOUR-GITHUB-USERNAME` above and in `.github/FUNDING.yml` with your real accounts before publishing — as placeholders they won't work.)*

## License

MIT — see [LICENSE](LICENSE).

## Honest framing

This is a genuine, correct architecture at a scale nobody has publicly
built specifically for code, as far as public information goes. The code
isn't fake or a toy — but reaching a *working* Fade is a data +
infrastructure + time problem, not a code problem, and at 3.1T parameters
that's a genuinely large undertaking regardless of funding.
