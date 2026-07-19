# Contributing to Fade

Thanks for your interest in contributing. This project is an open
architecture and training pipeline for a large-scale Mixture-of-Experts
code model — contributions of all sizes are welcome, from typo fixes to
infrastructure improvements.

## Ways to contribute

- **Bug reports** — open an issue with steps to reproduce
- **Architecture improvements** — e.g. alternative routing strategies,
  attention variants, efficiency improvements
- **Infrastructure** — expert-parallel / model-parallel training support
  (DeepSpeed-MoE, Megatron-Core integration) is a major open area
- **Documentation** — clearer setup instructions, tutorials, diagrams
- **Data tooling** — tokenization scripts, dataset cleaning utilities

## Development setup

```bash
git clone https://github.com/YOUR-USERNAME/fade-model.git
cd fade-model
pip install -r requirements.txt
python config.py   # sanity check the parameter count math, no GPU needed
```

## Pull request guidelines

1. Fork the repo and create a branch from `main`
2. Keep PRs focused — one logical change per PR is easier to review
3. Include a short description of what changed and why
4. If you change the architecture, run `python config.py` and confirm the
   parameter counts still make sense, and note the new numbers in your PR

## Code style

Standard PEP 8 for Python. No strict linter enforced yet — keep it
readable and consistent with the existing files.

## Questions

Open an issue for anything unclear — this project is early and
documentation will keep improving as it grows.
