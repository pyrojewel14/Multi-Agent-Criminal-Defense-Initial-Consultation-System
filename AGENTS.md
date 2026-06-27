# Project Rules

## Python Environment

When running any Python command in this project, use the conda environment named `rag`.

Prefer `conda run -n rag ...` so the environment is explicit in non-interactive commands, for example:

```bash
conda run -n rag python ...
conda run -n rag pytest ...
conda run -n rag uvicorn ...
```

For an interactive shell, activate the environment first:

```bash
conda activate rag
```

Do not run project Python code, tests, type checks, formatters, or backend services with the system Python, a global Python install, another conda environment, or a different virtual environment.
