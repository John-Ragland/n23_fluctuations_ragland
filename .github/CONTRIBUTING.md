See the [Scientific Python Developer Guide][spc-dev-intro] for a detailed
description of best practices for developing scientific packages.

[spc-dev-intro]: https://learn.scientific-python.org/development/

# Development environment

This package is developed inside the
[NESCAFE23](https://github.com/John-Ragland/NESCAFE23) workspace. Follow the
installation steps in the [README](../README.md#installation--setup), which
create one environment at the workspace root with this package installed in
editable mode, along with the test, dev and docs dependency groups.

# Pre-commit

Notebook outputs are stripped by `nbstripout`, and ruff runs as a pre-commit
hook. Enable both from within `projects/n23_fluctuations_ragland/`:

```bash
uv run nbstripout --install
uv run pre-commit install
```

You can also run `uv run pre-commit run --all-files` to check everything
without installing the hook.

# Testing

```bash
uv run pytest projects/n23_fluctuations_ragland/tests
```

# Coverage

```bash
uv run pytest projects/n23_fluctuations_ragland/tests --cov=n23_fluctuations_ragland
```

# Building docs

```bash
uv run sphinx-build -b html docs docs/_build/html
```
