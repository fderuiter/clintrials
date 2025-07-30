# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> **Note to maintainers:** Rename the version heading below (e.g., to `0.2.0` or `1.0.0`) when you tag the release.
> Date below reflects when this set of changes was prepared.

## [Unreleased] — 2025-07-30

### Added
- Quadrature back‑end for `prob_tox_exceeds`. (`8b101bb`)
- Adaptive 1‑D posterior integrator for improved accuracy/efficiency. (`b5db76b`)
- Tests for final decision path in TTE. (`41d1a62`)
- Coverage reporting via `pytest-cov`; coverage enforcement in CI. (`491fde7`, `d953a74`)
- Static placeholder in docs and initial `CHANGELOG`. (`21d5a81`, `cc71c44`)

### Changed
- **Packaging:** switch to **Poetry** for dependency management/builds. (`7f46346`, #1)
- **Testing:** migrate suite from **nose** to **pytest**. (`14fbbfa`)
- **CI matrix:** version‑specific dependencies and resolver tweaks; run on Python 3.9–3.12. (`e7a362d`, `4a79db8`, `da21f44`)
- **Docs site:** deploy Sphinx‑built HTML to GitHub Pages; bypass Jekyll so `_static/` is served (replaces earlier Jekyll‑based attempts). (`d8039c1`, `eca76be`, `8799252`)
- **Logging:** replace `print` statements with logging; add `NullHandler`. (`60c15dd`)
- **Style:** apply `black`, `isort`, and `ruff` across the codebase. (`9eddbb5`, `4fecefa`, `2aeca03`)
- **Numerical tests:** relax tolerances for EffTox v2 / Thall2014 to reduce brittleness. (`7d4c8e2`, `5d73efa`)
- **Docs content:** general modernization and polish; clarified integration behavior and defaults. (`3099b7a`, `d0b5e43`, `437c5d0`, `b29bca5`)
- **Ensure tests install local package** in CI to validate current changes. (`10e630c`)

### Fixed
- Import compatibility: use `collections.abc.Iterable` on modern Python. (`c986f7a`)
- Integration defaults and associated **Wages–Tait** tests. (`75f4c1a`, #5)
- Intersphinx configuration. (`c29a545`)
- Lint workflow by explicitly installing `pre-commit`. (`1b822de`)
- WATU tests and coverage stability. (`7ba9360`)
- Docs changelog link. (`b29bca5`)

### Removed
- **Wages–Tait:** remove `plugin_mean` parameter; behavior corresponds to the former `plugin_mean=False`. (`1f0e8bc`)
- Drop Python 3.8 from the test matrix. (`4a79db8`)
- Remove unused `conf_old.py`. (`2d1b3f2`, `fa7c4e0`)
- Remove previously added release workflow (semantic‑release). (`236ce82`, see also `a200a51` below)

### Security
- Addressed GitHub Advanced Security code‑scanning suggestions (tightened workflow permissions / hardening). (`9b1a169`, `227b363`)

### CI / Release Automation
- `semantic-release` workflow added then removed while revisiting release strategy. (`a200a51`, `236ce82`)
- Use Python 3.9 for the (now removed) release workflow. (`02fa474`)

### Documentation / Site Pipeline (context)
- Earlier: experimented with “deploy using jekyll” and `gh-pages` branch. (`be80966`, `4e444d8`, `2a9d99e`)
- Now: standardized on Sphinx build → Pages deploy with Jekyll bypassed; `_static/` assets are served correctly. (`d8039c1`, `eca76be`, `8799252`)

### Breaking changes
- **Minimum Python version is now 3.9.** (`4a79db8`)
- **Wages–Tait:** `plugin_mean` removed; callers relying on `plugin_mean=True` must update usage. (`1f0e8bc`)

### Migration notes
- **Development workflow** is Poetry‑based:
  ```bash
  poetry install
  poetry run pre-commit install
  poetry run pytest
  poetry run sphinx-build -b html docs docs/_build/html
  ```
- **Logging** now controls output; configure handlers/levels instead of relying on `print`.

### Merged PRs (representative)
- #19: Refactor oversized component(s) (merge commit `87f34a0`).
- #20: Replace normal approximations where appropriate (merge commit `9ad53c3`).
- #21: Fix biased posterior summaries (merge commit `397f99c`).
- #22: Modify workflow/docs for Jekyll (superseded by current Sphinx→Pages approach) (merge commit `2a9d99e`).
- #23: Remove unused `conf_old.py` (merge commit `fa7c4e0`).

---

_Commit references above use short SHAs from your history for traceability._
_Update the heading to the final version when cutting a release, and add a `[Unreleased]` compare link target as desired._

## v0.1.4 – initial release
