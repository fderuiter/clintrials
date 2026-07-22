# Contributing Guidelines

Welcome, and thank you for your interest in contributing to the project! We rely on our community to keep this repository healthy, secure, and well-maintained.

## How to Contribute

1. Check our `ROADMAP.md` for prioritized maintenance tasks, technical debt, and architectural priorities.
2. Search through existing GitHub issues before creating a new one.
3. When creating an issue (e.g., bug report, feature request, or maintenance task), please use the provided issue templates and reference relevant `ROADMAP.md` milestones.

## Core Maintenance and Structural Refactoring

Tackling structural technical debt is critical for the long-term health of our project. `ROADMAP.md` provides clear strategic direction on architectural and security priorities, allowing you to select high-impact work.

### High-Complexity Core Modules

We require standardized approaches when dealing with high-complexity code, especially in the core recruitment and dose-finding modules. This ensures that structural changes and logic transitions follow project standards without requiring constant oversight.

#### Recruitment Logic (`clintrials/core/recruitment.py`)
- **Refactoring Complex Classes**: The recruitment module manages complex simulation streams (e.g., `QuadrilateralRecruitmentStream`). When making structural transitions or renaming components:
  - Break down massive classes into smaller, composable units.
  - Maintain backward compatibility of external interfaces and APIs during transitions.
  - Thoroughly document and test logic changes.

#### Dose-Finding Logic (`clintrials/dosefinding/crm.py`)
- **Numerical Stability & Best Practices**: When modifying statistical or probability models in this module:
  - **Avoid Anti-Patterns**: Do not use loops combined with linear product calculations (e.g., `np.prod`) for likelihoods. Use mathematically stable, vectorized operations.
  - **Vectorization**: Compute likelihoods using logarithmic spaces (e.g., `np.log` and `np.sum`) to prevent numerical instability, underflow, or overflow.
  - **DRY Principles**: Avoid repeating hardcoded grids (like `np.linspace(-5, 5, 1000)`). Centralize them as constants.

### Dependency Upgrades & Security

- **Security First**: High-risk security dependencies listed in `ROADMAP.md` are top priorities.
- **Handling Vulnerabilities**: Ensure dependency updates are tested thoroughly, and adhere to standard disclosure policies before detailing new vulnerabilities in the public roadmap.

## Test-Driven Development (TDD) Sequencing

This project enforces strict Test-Driven Development (TDD) practices for all new business logic. Our Continuous Integration (CI) pipeline includes a Git History Timeline Auditor that verifies the order of commits in your pull request.

- **TDD Sequencing Requirements**: Tests must be written before or alongside the business logic they verify. The CI pipeline ensures that any commit modifying a source file is preceded by, or accompanied by, a commit that adds or modifies the corresponding test file.
  - The corresponding test file for a module like `clintrials/core/recruitment.py` should be named `test_recruitment.py` or `test_recruitment_*.py` and placed in the `tests/` directory.

- **Urgent Hotfix Exemptions**: If you are deploying an urgent hotfix to production and cannot adhere to TDD sequencing rules, you can bypass the timeline audit using one of the following methods:
  - **Branch Name Prefix**: Prefix your pull request branch name with `hotfix/` (e.g., `hotfix/urgent-bug-fix`).
  - **Git Commit Trailer**: Append the `skip-tdd` trailer to any commit message in your pull request branch (e.g., by adding `skip-tdd: true` or `skip-tdd: security-hotfix` on a new line at the bottom of the commit message).

## Pull Request Process
- Ensure all tests pass.
- Link your pull request to the relevant issue.
- Describe your changes clearly in the pull request description.

### Documentation Verification
When modifying documentation files (Markdown or reStructuredText), please ensure that all internal repository links and file paths are valid. You can run the automated documentation path-validation test locally using the following command:

```bash
poetry run pytest tests/test_docs_links.py
```

## Promoting Features to the Public API

If you develop a core utility (e.g., a numerical integration method or a math function) that would be useful for researchers outside of the internal modules, you can promote it to the public API surface.

To transition a feature from an internal utility to public status, follow these steps:
1. Ensure the function or class has a complete docstring that clearly describes its purpose, arguments, and return values. This is required to pass existing linting rules and ensure it is properly rendered in the documentation.
2. Ensure the utility is not an internal-only helper. Internal-only helpers must remain hidden using standard naming conventions (e.g., prefixing with an underscore `_`) to avoid cluttering the public API.
3. Import the newly promoted utility in the package root `clintrials/__init__.py`.
4. Add the utility to the `__all__` list in `clintrials/__init__.py`.
5. Update the API documentation index at `README.md` to include the new top-level utility so it is discoverable by users.
6. Run `poetry run python scripts/verify_api_signatures.py --generate` to update the `api_manifest.json` file. Commit this file along with your changes.
