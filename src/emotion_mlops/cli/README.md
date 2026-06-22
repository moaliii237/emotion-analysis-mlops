# EmotionScope CLI

Production-ready command-line interface for the EmotionScope emotion classification API. Built with Typer for type-safe argument parsing and Rich for colourful terminal output.

## 1. What is the CLI?

A way to use the EmotionScope model from your terminal, no browser, no web frontend, just a single command.

```bash
emotionscope predict "I'm so happy today!"
```

This command calls your local FastAPI backend over HTTP and prints a coloured bar chart with the prediction result.

## 2. Why Typer?

Typer (by the FastAPI author, Sebastián Ramírez) is the modern Python CLI framework:
- Type hints become automatic input validation
- Built-in Rich integration for coloured output
- Auto-generated `--help` documentation from docstrings
- Native subcommand support (`predict`, `health`, `pipeline`)

## 3. Folder structure

```
src/emotion_mlops/cli/
  __init__.py        # Package init
  main.py            # Typer app entry point + routing
  commands.py        # Modern Typer-based commands
  infer.py           # Legacy argparse pipeline driver (Sprint 2)
  README.md          # This file
```

Separation of concerns:
- `main.py`: routing only (command names to functions)
- `commands.py`: the actual work (format output, write to terminal)
- `infer.py`: API call logic (HTTP requests)

## 4. The four commands

### `predict`, single-sentence prediction

```bash
emotionscope predict "I am so excited about this!"
```

Returns the top emotion + confidence as a coloured bar chart.

### `predict-full`, full 7-emotion distribution

```bash
emotionscope predict-full "I am happy"
```

Returns probabilities for all 7 emotion labels. Used for the radar chart on the frontend.

### `health`, backend status check

```bash
emotionscope health
```

Verifies API connectivity and reports the loaded model version.

### `pipeline`, full audio/CSV pipeline

```bash
emotionscope pipeline --file audio.mp3
```

Runs Whisper transcribe, then MarianMT translate, then DeBERTa classify, and saves the result as CSV.

## 5. Why two CLI styles?

- `commands.py` (Typer) is the modern interface: type hints, colours, subcommands, JSON output
- `infer.py` (argparse) is the legacy pipeline driver from Sprint 2 and remains available via `python -m emotion_mlops.cli.infer ...` for backwards compatibility. The new `emotionscope pipeline` command delegates to it so we don't duplicate logic

## 6. Shell completion (Typer bonus feature)

Typer can install tab completion scripts for your shell:

```bash
# Install for the current shell
emotionscope --install-completion

# Or just print the script
emotionscope --show-completion
```

After installation, restart your shell. Now `emotionscope <TAB>` will auto-suggest commands and options.

## 7. Use in scripts

The `--json` output of `predict-full` is designed for piping:

```powershell
# PowerShell
$result = emotionscope predict-full "I am angry" --json | ConvertFrom-Json
$result.top_emotion        # anger
$result.top_confidence     # 0.92
```

```bash
# Bash / WSL
emotionscope predict-full "I am angry" --json | jq '.top_emotion'
# "anger"
```

## 8. Demo cheat sheet

Five commands to run during the demo:

```bash
# 1. Show the help menu
emotionscope --help

# 2. Health check
emotionscope health

# 3. Quick prediction
emotionscope predict "Trump shouted at the journalists"

# 4. Verbose prediction
emotionscope predict "Mary Berry smiled warmly" --verbose

# 5. Full distribution as a bar chart
emotionscope predict-full "I'm both excited and scared"
```

## 9. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `emotionscope: command not found` | Package not installed in editable mode | `pip install -e . --no-deps` |
| `ModuleNotFoundError: emotion_mlops.cli.command` | `pyproject.toml` points at wrong filename | Check `pyproject.toml`, `cli.main:main` |
| `AttributeError: 'EmotionClassifier' object has no attribute 'predict_all'` | Older classifier, `predict_all` not yet added | Add `predict_all()` method to `models/classifier.py` |
| First command takes ~20 s | DeBERTa model loading from disk | Normal, subsequent commands reuse the loaded model |
| No colours in output | Terminal doesn't support ANSI | Use PowerShell 7+ or Windows Terminal |

## 10. Sprint mapping

| ILO | How the CLI contributes |
|---|---|
| **ILO 9.3A** (clean code & packaging), Sprint 2 | Provides a `console_script` entry point with type-hinted, documented commands |
| **ILO 9.3B** (documentation), Sprint 2 | Every command carries a docstring + `--help`, plus this README |

Last updated: Sprint 4, Block D, BUas NLP Group 1, Ceren Kurt