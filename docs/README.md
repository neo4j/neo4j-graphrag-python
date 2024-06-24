# Sphinx Documentation

Building the docs requires Python 3.9+

Ensure the dev dependencies in `pyproject.toml` are installed.

From the root directory, run the Makefile:

```
make -C docs html
```

You can now view your build locally.

When you open a PR, a TeamCity build will trigger when it's ready and someone from Neo4j
can then preview this.
