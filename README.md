# dlinit

CLI to bootstrap reproducible PyTorch/NLP project folders.

It can:
- Create a clean project template (data/resources/notebooks/src/checkpoints/training_summary/scripts)
- Create a `.venv` with built-in `venv`
- Install a minimal stack: numpy, pandas, scikit-learn, torch, transformers, matplotlib, tqdm, nltk
- Generate a templated Jupyter notebook (title + created date + aim + conclusion + starter imports)

## Install

### Recommended (pipx)
```bash
pipx install git+https://github.com/<your-username>/<repo>.git
