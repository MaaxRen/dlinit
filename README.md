# dlinit

CLI to create reproducible NLP deep learning project folders.

It can:
- Create a clean project template (data/resources/notebooks/src/checkpoints/training_summary/scripts)
- Create a `.venv` with built-in `venv`
- Install a minimal stack: numpy, pandas, scikit-learn, torch, transformers, matplotlib, tqdm, nltk
- Generate a templated Jupyter notebook (title + created date + aim + conclusion + starter imports)

## Install

### Recommended (pipx)
```bash
pipx install git+https://github.com/MaaxRen/dlint.git
```

### pip
```bash
pip install git+https://github.com/MaaxRen/dlint.git
```

### Development (editbale)
```bash
git clone https://github.com/MaaxRen/dlint.git
cd dlinit
python -m pip install -e .
```

## Usage
### Project Template (`dlinit`)
#### Create a new project (in the current folder)
```bash
cd /path/where/you/store/projects
dlinitn <project_name>
```

#### Create a new project (somewhere else)
```bash
dlinit <project_name> --path /path/where/you/store/projects
```

#### Skip dependency installation
```bash
dlinit <project_name> --no-install
```

#### Typical layout
```bash
<project_name>/
  data/raw
  data/processed
  resources/prompts
  resources/configs
  resources/data_models
  resources/papers
  notebooks
  src/<project_name>/utils.py
  model_checkpoints
  training_summary/metrics
  training_summary/plots
  training_summary/notes
  scripts
  .venv/            (after init)
  requirements.lock (after install)
```

### Notebook Template (`dlnb`)
#### Create a new notebook (in the current folder):
```bash
cd /path/to/<project_name>/notebooks
dlnb
```

#### Create a new notebook (somewhere else)
```bash
dlnb --dir /path/to/<project_name>/notebooks
```

You will be prompted to provide a title, can otherwise provide title non-interactively:
``` bash
dlnb --title "Exploratory Data Analysis"
```

### Experiment Note Template (`dlnote`)
Creates a simple markdown note intended as quick “ground truth” for an LLM to read and generate richer summaries.

#### Create a new note (recommended location)
```bash
cd /path/to/notes
dlnote
```

By default, if ```training_summary/notes``` exists, the note will be created there.

#### Create a new note (somewhere else)
```bash
dlnote --dir /path/to/<project_name>
```

You will be prompted to provide a title, can otherwise provide title non-interactively:
``` bash
dlnote --title "Used a different model"
```

Optionally set the output filename (without ```.md```):
```bash
dlnote --name "new_model_try1"
```

#### Typical Note Structure
The generated note contains:
- Hypothesis
- Change
- Result (metrics + qualitative)
- Next
- Links (checkpoints/plots/logs)

## License
MIT (see ```LICENSE```)







