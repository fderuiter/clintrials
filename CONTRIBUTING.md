# Contributing

Contributions are welcome! We value contributions from the community and are excited to see how you can help improve clintrials.

## How to Contribute

There are many ways to contribute to clintrials, including:

- Reporting bugs
- Suggesting enhancements
- Improving documentation
- Submitting pull requests

### Reporting Bugs

If you find a bug, please open an issue on our GitHub repository. When you report a bug, please include the following information:

- A clear and descriptive title
- A detailed description of the problem, including steps to reproduce the bug
- The version of clintrials you are using
- Your Python version
- Any relevant error messages or tracebacks

### Suggesting Enhancements

If you have an idea for a new feature or an improvement to an existing one, please open an issue on our GitHub repository. Please provide a clear and detailed description of your suggestion and why you think it would be a valuable addition to the project.

### Pull Request Process

1.  Fork the repository and create your branch from `main`.
2.  Install the development dependencies:
    ```bash
    poetry install
    ```
3.  Make your changes and add tests for them.
4.  Ensure the test suite passes:
    ```bash
    poetry run pytest
    ```
5.  Make sure your code follows the existing code style. We use `black`, `ruff`, and `isort` to enforce a consistent code style. You can run the linters with:
    ```bash
    poetry run black .
    poetry run ruff .
    poetry run isort .
    ```
6.  Commit your changes and push your branch to your fork.
7.  Open a pull request to the `main` branch of the clintrials repository.

## Code of Conduct

Please note that this project is released with a Contributor Code of Conduct. By participating in this project you agree to abide by its terms. We are committed to fostering a welcoming and inclusive community.
