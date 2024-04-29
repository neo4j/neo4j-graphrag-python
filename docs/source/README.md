# Sphinx Documentation

Building the docs requires Python 3.8.1+

Ensure the dev dependencies in `pyproject.toml` are installed.

From the root directory
```
make -C docs html
```

```
python -m sphinx -b html docs/source docs/build/html
```
