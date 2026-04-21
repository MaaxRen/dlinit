# dlworkflow

CLI to enhance efficiency, traceability, and reproducibility in a deep learning / machine learning workflow.

It can:
- Create a clean project directory with a Python virtual environment and starter ML packages
- Remember general CLI settings such as time zone and default profile
- Generate a lightweight templated Jupyter notebook
- Generate a lightweight templated experiment note
- Capture structured training metadata from a training script
- More to be developed, including profile-based initialization

## Install

### Recommended (pipx)
```bash
pipx install git+https://github.com/MaaxRen/dlworkflow.git
```

### pip
```bash
pip install git+https://github.com/MaaxRen/dlworkflow.git
```

### Development (editable)
```bash
git clone https://github.com/MaaxRen/dlworkflow.git
cd dlworkflow
python -m pip install -e .
```

## Usage
### Project Template (`dlinit`)
`dlinit` uses the saved `default_profile` from `dlsetup`. If no default profile has been configured yet, it falls back to `ML`.

#### Create a new project (in the current folder)
```bash
cd /path/where/you/store/projects
dlinit <project_name>
```

#### Create a new project (somewhere else)
```bash
dlinit <project_name> --path /path/where/you/store/projects
```

#### Skip dependency installation
```bash
dlinit <project_name> --no-install
```

#### Profiles
The starter dependency set depends on the active profile:

- `ML`: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `tqdm`, `jupyter`, `ipykernel`
- `NLP`: `numpy`, `pandas`, `scikit-learn`, `torch`, `transformers`, `datasets`, `tokenizers`, `nltk`, `matplotlib`, `seaborn`, `tqdm`, `jupyter`, `ipykernel`
- `CV`: `numpy`, `pandas`, `scikit-learn`, `torch`, `torchvision`, `opencv-python`, `pillow`, `albumentations`, `matplotlib`, `seaborn`, `tqdm`, `jupyter`, `ipykernel`
- `STAT`: `numpy`, `pandas`, `scikit-learn`, `scipy`, `statsmodels`, `matplotlib`, `seaborn`, `jupyter`, `ipykernel`

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
  src/<python_package_name>/utils.py
  model_checkpoints
  training_summary/runs
  training_summary/plots
  training_summary/notes
  scripts
  .venv/            (after init)
  requirements.lock (after install)
```

### General Setup (`dlsetup`)
Stores general settings that should be reused across commands. It is intended to grow as more reusable CLI configuration is added.

Set a default time zone:
```bash
dlsetup --timezone Australia/Adelaide
```

Set a default profile:
```bash
dlsetup --default_profile NLP
```

Show the saved configuration:
```bash
dlsetup --show
```

Clear a saved setting:
```bash
dlsetup --clear timezone
dlsetup --clear default_profile
```

The saved time zone is used by generated notebook dates, note timestamps, and experiment logging timestamps unless you override it in code. The saved `default_profile` is intended for future profile-aware commands such as `dlinit`.
`dlinit` now consumes `default_profile` directly and uses `ML` if no profile has been saved yet.

### Notebook Template (`dlnb`)
Creates a notebook with:
- title
- created date
- aim section
- conclusion section
- starter imports
- autoreload setup

The generated notebook places the cells in this order:
- title and notebook context markdown
- starter imports
- autoreload setup

#### Create a new notebook (in the current folder)
```bash
cd /path/to/<project_name>/notebooks
dlnb
```

#### Create a new notebook (somewhere else)
```bash
dlnb --dir /path/to/<project_name>/notebooks
```

You will be prompted to provide a title, can otherwise provide title non-interactively:
```bash
dlnb --title "Exploratory Data Analysis"
```

### Experiment Note Template (`dlnote`)
Creates a lightweight markdown note intended as quick ground truth for an LLM to read and generate richer summaries.

#### Create a new note (recommended location)
```bash
cd /path/to/<project_name>
dlnote
```

If `training_summary/notes` exists under the detected project root, the note will be created there by default.

#### Create a new note (somewhere else)
```bash
dlnote --dir /path/to/<project_name>
```

You will be prompted to provide a title, can otherwise provide title non-interactively:
```bash
dlnote --title "Used a different model"
```

Optionally set the output filename (without `.md`):
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

### Training Metadata Logging
The package also includes a small helper and decorator for recording structured run metadata from a training script.

Direct helper example:
```python
from dlworkflow import save_training_metadata

MODEL_NAME = "distilbert-base-uncased"
BATCH_SIZE = 32
LEARNING_RATE = 3e-5
NUM_EPOCHS = 5
BEST_VAL_F1 = 0.842
TRAINING_NOTE = "Added class weighting and lower learning rate."

paths = save_training_metadata(globals())
print(paths["json_path"])
```

Decorator example:
```python
from dlworkflow import log_training_run

MODEL_NAME = "distilbert-base-uncased"
BATCH_SIZE = 32
LEARNING_RATE = 3e-5
NUM_EPOCHS = 5

@log_training_run(filename="baseline")
def train_model():
    best_val_f1 = 0.842
    return {"best_val_f1": best_val_f1}

train_model()
```

This writes:
- one JSON file per run in `training_summary/runs/`
- an append-only `training_summary/runs/runs.jsonl` file for aggregation

The logging utilities collect ALL-CAPS variables from the training module, convert common `numpy`, `torch`, and `Path` values into JSON-safe forms, and add run timestamps automatically. The decorator also records `status`, `duration_sec`, `function_name`, and any returned value under `result`. If the wrapped function raises an exception, the failure is logged before the exception is re-raised.
If you have configured a time zone with `dlsetup`, the logger uses that by default.

## Notes
- `dlinit` derives the Python package name from the project name, converting spaces and hyphens to underscores and removing invalid characters.
- `dlnb` writes into `notebooks/` if that directory exists under the target directory; otherwise it writes into the provided directory.
- `dlnote` avoids overwriting existing notes by adding a numeric suffix when needed.

## License
MIT (see `LICENSE`)
