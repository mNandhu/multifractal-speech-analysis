---
description: When adding/removing/modifying a dependency to the current venv, or running a python file
# applyTo: 'Describe when these instructions should be loaded' # when provided, instructions will automatically be added to the request context when the pattern matches an attached file
---

Whenever you add/remove/modify a dependency in the current virtual environment, or run a python file, please use `uv` to run the command instead of `pip` or `python`. This ensures that the correct virtual environment is activated and the dependencies are managed properly. For example, use `uv add <package>` to add a package to pyproject.toml, and `uv run python <file.py>` to run a python file.
