# Contributing

Contributions are welcome! Clone the repository and install the development dependencies:

```bash
pip install -e .[docs]
```

Run the test suite with `pytest` and build the docs using `make -C docs html`.

Please follow the existing code style enforced by **black**, **ruff**, and **isort**.

## Promoting Features to the Public API

If you develop a core utility (e.g., a numerical integration method or a math function) that would be useful for researchers outside of the internal modules, you can promote it to the public API surface.

To transition a feature from an internal utility to public status, follow these steps:
1. Ensure the function or class has a complete docstring that clearly describes its purpose, arguments, and return values. This is required to pass existing linting rules and ensure it is properly rendered in the documentation.
2. Ensure the utility is not an internal-only helper. Internal-only helpers must remain hidden using standard naming conventions (e.g., prefixing with an underscore `_`) to avoid cluttering the public API.
3. Import the newly promoted utility in the package root `clintrials/__init__.py`.
4. Add the utility to the `__all__` list in `clintrials/__init__.py`.
5. Update the API documentation index at `docs/reference/index.rst` to include the new top-level utility so it is discoverable by users.
