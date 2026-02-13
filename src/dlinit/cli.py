from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from pathlib import Path
import re
import json
from datetime import datetime

FOLDERS = [
    "data/raw",
    "data/processed",
    "resources/prompts",
    "resources/configs",
    "resources/data_models",
    "resources/papers",
    "notebooks",
    "src",
    "model_checkpoints",
    "training_summary/metrics",
    "training_summary/plots",
    "training_summary/notes",
    "scripts",
]

REQUIREMENTS = [
    "numpy",
    "pandas",
    "scikit-learn",
    "torch",
    "transformers",
    "matplotlib",
    "tqdm",
    "nltk",
]

GITIGNORE = """\
# Python
__pycache__/
*.py[cod]
*.pyd
*.so
*.egg-info/
dist/
build/

# Envs
.venv/
venv/
ENV/

# Notebooks
.ipynb_checkpoints/

# OS
.DS_Store
Thumbs.db

# Data / artifacts (keep structure, not contents)
data/**/*
!data/**/.gitkeep

model_checkpoints/**/*
!model_checkpoints/**/.gitkeep

training_summary/**/*
!training_summary/**/.gitkeep
"""

NOTE_TMPL = """\
# {title}

Created: {created}

## Hypothesis
- 

## Change
- 

## Result
- Metrics:
- Qualitative:

## Next
- 

## Links
- checkpoints:
- plots:
- logs:
"""

def run(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def venv_paths(project_dir: Path) -> tuple[Path, Path]:
    venv_dir = project_dir / ".venv"
    if platform.system().lower().startswith("win"):
        py = venv_dir / "Scripts" / "python.exe"
        pip = venv_dir / "Scripts" / "pip.exe"
    else:
        py = venv_dir / "bin" / "python"
        pip = venv_dir / "bin" / "pip"
    return py, pip

def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def touch_gitkeep(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / ".gitkeep").write_text("", encoding="utf-8")


def create_structure(project_dir: Path) -> None:
    for rel in FOLDERS:
        (project_dir / rel).mkdir(parents=True, exist_ok=True)

    # Track empty dirs in git
    for rel in ["data/raw", "data/processed", "model_checkpoints", "training_summary"]:
        touch_gitkeep(project_dir / rel)

    write_text(project_dir / ".gitignore", GITIGNORE)

    pkg_name = project_to_pkg_name(project_dir.name)
    pkg_dir = project_dir / "src" / pkg_name
    pkg_dir.mkdir(parents=True, exist_ok=True)

    write_text(pkg_dir / "__init__.py", "")
    write_text(pkg_dir / "utils.py", "")


def create_venv(project_dir: Path, python: str) -> None:
    run([python, "-m", "venv", str(project_dir / ".venv")])


def install_deps(project_dir: Path) -> None:
    py, _ = venv_paths(project_dir)
    if not py.exists():
        raise FileNotFoundError("Venv python not found. Did venv creation fail?")

    run([str(py), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    run([str(py), "-m", "pip", "install", *REQUIREMENTS])

    lock_path = project_dir / "requirements.lock"
    frozen = subprocess.check_output([str(py), "-m", "pip", "freeze"], text=True)
    lock_path.write_text(frozen, encoding="utf-8")

def project_to_pkg_name(project_name: str) -> str:
    """
    Convert a project folder name like 'domain-adaptation' into a valid
    Python package name like 'domain_adaptation'.
    """
    name = project_name.strip().lower()
    name = re.sub(r"[\s\-]+", "_", name)          # spaces/hyphens -> underscore
    name = re.sub(r"[^a-z0-9_]", "", name)        # drop other chars
    name = re.sub(r"_+", "_", name).strip("_")    # collapse underscores
    if not name:
        name = "project"
    if name[0].isdigit():
        name = f"p_{name}"
    return name

def slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = s.strip("-")
    return s or "note"

def find_project_root(start: Path) -> Path:
    cur = start.resolve()
    for parent in [cur, *cur.parents]:
        if (parent / "src").is_dir():
            return parent
    return cur

def write_note_md(path: Path, title: str, project: str) -> None:
    created = datetime.now().strftime("%Y-%m-%d %H:%M")
    content = NOTE_TMPL.format(title=title, created=created, project=project)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")

def write_ipynb(path: Path, title: str, pkg_name: str) -> None:
    created = datetime.now().strftime("%Y-%m-%d")
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"# {title}\n\n",
                    f"**Created:** {created}\n\n",
                    "## Aim\n",
                    "- TODO\n",
                    "- What is the aim of this notebook?\n\n",
                    "## Conclusion\n",
                    "- TODO\n",
                    "- What have you concluded?\n",
                    "- What recommendation/insights can you give?"
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import os\n",
                    "import json\n",
                    "import numpy as np\n",
                    "import pandas as pd\n",
                    "import matplotlib.pyplot as plt\n",
                    "\n",
                    "# Allow imports from ./src (src-layout)\n",
                    "import sys\n",
                    "from pathlib import Path\n",
                    "sys.path.insert(0, str(Path.cwd().parent / 'src'))\n",
                ],
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(nb, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dlinit",
        description="Initialize a torch/NLP project directory with a venv and minimal deps.",
    )
    parser.add_argument("project_name", help="Directory name to create (or existing empty directory).")
    parser.add_argument("--path", default=".", help="Parent path (default: current directory).")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to create venv (default: current python).",
    )
    parser.add_argument("--no-install", action="store_true", help="Skip installing dependencies.")
    args = parser.parse_args(argv)

    parent = Path(args.path).expanduser().resolve()
    project_dir = (parent / args.project_name).resolve()
    project_dir.mkdir(parents=True, exist_ok=True)

    # Safety: refuse to overwrite non-empty dirs (unless it looks like the template)
    if any(project_dir.iterdir()):
        allowed = {".gitignore", "README.md", ".venv", "requirements.lock"} | {p.split("/")[0] for p in FOLDERS}
        existing = {p.name for p in project_dir.iterdir()}
        if not existing.issubset(allowed):
            print(f"Error: {project_dir} is not empty. Refusing to overwrite.", file=sys.stderr)
            return 2

    create_structure(project_dir)

    if not (project_dir / ".venv").exists():
        create_venv(project_dir, args.python)

    if not args.no_install:
        install_deps(project_dir)

    py, _ = venv_paths(project_dir)
    print(f"Initialized: {project_dir}")
    print(f"Activate (mac/linux): source {project_dir}/.venv/bin/activate")
    print(f"Activate (windows):   {project_dir}\\.venv\\Scripts\\activate")
    print(f"Python: {py}")
    return 0

def notebook_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dlnb",
        description="Create a templated Jupyter notebook in the current folder (or a target folder).",
    )
    parser.add_argument("--title", default=None, help="Notebook title. If omitted, you will be prompted.")
    parser.add_argument(
        "--dir",
        default=".",
        help="Directory to create the notebook in (default: current directory).",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Notebook filename (without .ipynb). Default is <slugified-title>.",
    )
    args = parser.parse_args(argv)

    title = args.title or input("Notebook title: ").strip()
    if not title:
        print("Error: title is required.", file=sys.stderr)
        return 2

    slug = slugify(title)
    fname = (args.name or slug) + ".ipynb"

    target_dir = Path(args.dir).expanduser().resolve()

    # Determine project root for package name:
    # If notebooks/ exists, assume target_dir is project root; otherwise still use folder name.
    project_root = target_dir
    pkg_name = project_to_pkg_name(project_root.name)

    nb_dir = project_root / "notebooks"
    out_dir = nb_dir if nb_dir.exists() else project_root
    out_path = out_dir / fname

    write_ipynb(out_path, title, pkg_name)
    print(f"Created: {out_path}")
    return 0

def note_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dlnote",
        description="Create a templated experiment note markdown file (LLM-readable ground truth).",
    )
    parser.add_argument("--title", default=None, help="Note title. If omitted, you will be prompted.")
    parser.add_argument("--dir", default=".", help="Directory to create the note from (default: current directory).")
    parser.add_argument("--name", default=None, help="Filename (without .md). Default: slugified title.")
    args = parser.parse_args(argv)

    title = (args.title or input("Note title: ").strip())
    if not title:
        print("Error: title is required.", file=sys.stderr)
        return 2

    target_dir = Path(args.dir).expanduser().resolve()
    project_root = find_project_root(target_dir)

    # Prefer <project_root>/training_summary/notes if present; else use target_dir
    preferred = project_root / "training_summary" / "notes"
    out_dir = preferred if preferred.exists() else target_dir

    slug = slugify(title) or "note"
    base = args.name or slug
    out_path = out_dir / f"{base}.md"

    # Avoid overwrite by suffixing
    i = 2
    while out_path.exists():
        out_path = out_dir / f"{base}_{i}.md"
        i += 1

    write_note_md(out_path, title=title, project=project_root.name)
    print(f"Created: {out_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
