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

### Documentation Guidelines and Build Verification
For detailed guidelines on setting up and running our dual-build documentation pipelines, please refer to our dedicated [Robust Documentation Guide](/docs/DOCUMENTATION_GUIDE.md).

We support two documentation systems:
1. **Sphinx Pipeline:** Built from reStructuredText (`.rst`) files, useful for traditional API and manual structure.
2. **Custom Node Pipeline:** Compiles modern MDX/Markdown (`.md`/`.mdx`) files under `docs/reference` into a fast, fully searchable static site under `docs/dist`.

When modifying documentation files (Markdown or reStructuredText), please ensure that all internal repository links and file paths are valid. You can run the automated documentation path-validation test locally using the following command:

```bash
poetry run pytest tests/test_docs_links.py
```

## Git Rebasing and Conflict Resolution

When working on a feature branch, you may need to update your branch with the latest changes from the `main` branch. We recommend using `git rebase` to maintain a clean, linear commit history. Follow this structured process:

### 1. Assess and Synchronize the Base State
- **Analyze:** Before initiating any integration, confirm your local working environment is safe. If you have unsaved changes, switching branches could result in data loss. Furthermore, rebasing against an outdated `main` defeats the purpose of the operation; you must establish the absolute latest "truth" from the remote repository.
- **Execute:** Run `git status` to ensure a clean working tree. If clean, run `git checkout main` followed by `git pull origin main`.
- **Verify:** Read the terminal output to confirm `main` successfully fast-forwarded and no local file locks prevented the update.

### 2. Prepare the Feature Branch for History Rewriting
- **Analyze:** Switch back to your specific context. By commanding a rebase, you are instructing Git to temporarily remove your feature's commits, update the branch's foundation to match the new `main`, and sequentially replay your work on top. Mental preparation is key: this process may halt if Git cannot automatically reconcile your logic with the new base.
- **Execute:** Run `git checkout <your-feature-branch>`, then run `git rebase main`.
- **Verify:** Observe the terminal output. Confirm whether it says "Successfully rebased" or "Merge conflict" to determine your immediate next action.

### 3. Analyze, Synthesize, or Remake
- **Analyze:** If Git suspends the operation due to conflicts, assess the scale of the divergence. Ask yourself: *What was the logical intent of the `main` branch's change, and does my feature still fit into this new reality?* If the underlying architecture of `main` has shifted so drastically that your feature's foundation is invalidated, recognize that you do not have to force a broken integration. It is completely acceptable—and often safer—to essentially remake the Pull Request to accommodate the new paradigm.
- **Execute:**
  - *If remaking the PR:* Run `git rebase --abort`. Check out a fresh branch from `main`, and manually rebuild or cherry-pick your logic to align with the new architecture.
  - *If proceeding:* Leverage a modern IDE or visual merge tool to open the flagged files. Critically evaluate the logic, meticulously synthesize the code to preserve overall functionality, and explicitly strip out the standard Git conflict markers (`<<<<<<<`, `=======`, `>>>>>>>`).
- **Verify:** Before closing the file, mentally (or physically via local linting/testing) run the code to ensure you haven't created a syntax error or broken the business logic during the synthesis.

### 4. Confirm Resolution and Advance the Sequence
- **Analyze:** Assuming you proceeded with the rebase and have saved the synthesized files, explicitly inform Git that human intervention is complete for this specific commit. Staging the files acts as your confirmation mechanism. Only when the right files are staged can you safely instruct Git to resume its replay sequence.
- **Execute:** Run `git status` to see the modified files, stage them with `git add .`, and trigger the next phase with `git rebase --continue`.
- **Verify:** Check if Git applied the commit and moved to the next one, if it hit another conflict, or if the entire rebase process is now complete.

### 5. Safely Override the Remote History
- **Analyze:** Once the rebase has successfully completed locally, you have rewritten the commit history. Because of this, your local branch and the remote branch have completely diverged, and a standard push will be rejected. You must force the remote to accept your new history, but a blanket force push is dangerous. Use a "lease" to ensure you only overwrite the remote if no one else has pushed new work to your feature branch while you were rebasing.
- **Execute:** Run `git push origin <your-branch> --force-with-lease`.
- **Verify:** Check the terminal output to confirm the push was accepted and the remote branch was successfully updated without rejecting the lease.

## Promoting Features to the Public API

If you develop a core utility (e.g., a numerical integration method or a math function) that would be useful for researchers outside of the internal modules, you can promote it to the public API surface.

To transition a feature from an internal utility to public status, follow these steps:
1. Ensure the function or class has a complete docstring that clearly describes its purpose, arguments, and return values. This is required to pass existing linting rules and ensure it is properly rendered in the documentation.
2. Ensure the utility is not an internal-only helper. Internal-only helpers must remain hidden using standard naming conventions (e.g., prefixing with an underscore `_`) to avoid cluttering the public API.
3. Import the newly promoted utility in the package root `clintrials/__init__.py`.
4. Add the utility to the `__all__` list in `clintrials/__init__.py`.
5. Update the API documentation index at `README.md` to include the new top-level utility so it is discoverable by users.
6. Run `poetry run python scripts/verify_api_signatures.py --generate` to update the `api_manifest.json` file. Commit this file along with your changes.
