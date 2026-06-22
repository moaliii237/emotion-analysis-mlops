"""
Command-line entry point for the EmotionScope package.

When installed via `pip install -e .`, this module is exposed as the
`emotionscope` console script (see pyproject.toml's `[project.scripts]`).
Users can then call commands like `emotionscope predict "..."` directly.
"""

from emotion_mlops.cli.commands import app


def main():
    """Launch the Typer CLI app."""
    app()


if __name__ == "__main__":
    main()
