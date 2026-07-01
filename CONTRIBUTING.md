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

## Pull Request Process
- Ensure all tests pass.
- Link your pull request to the relevant issue.
- Describe your changes clearly in the pull request description.
