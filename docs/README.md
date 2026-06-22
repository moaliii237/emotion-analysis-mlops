<<<<<<< HEAD
# EmotionScope Documentation

This directory contains the Sphinx-generated documentation for the EmotionScope package.

## Requirements

Make sure you have the required packages installed:

```bash
pip install sphinx sphinx-autodoc-typehints furo
```

## Generating the Documentation

From the root of the project, run:

```bash
sphinx-apidoc -o docs/source src/emotion_mlops
docs\make.bat html
```

## Viewing the Documentation

After building, open the following file in your browser:
## Structure

- `docs/source/conf.py`, Sphinx configuration file
- `docs/source/index.rst`, Main documentation entry point
- `docs/source/*.rst`, Auto-generated module pages
- `docs/build/html/`, Generated HTML documentation site
=======
## Command Line Interface (CLI)

This project provides a command-line interface (CLI) that allows users to run the emotion analysis pipeline without writing any Python code.

The CLI supports both transcript-based and audio/video-based inference.

---

## Installation

Install dependencies and the package:

```bash
pip install -r requirements.txt
pip install -e .


>>>>>>> 1f26e0c3b61cd45ff95a8b7c927daa2685c0b95e
