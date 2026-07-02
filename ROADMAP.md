# Technical Debt Roadmap

This roadmap tracks structural technical debt, high-priority maintenance tasks, and core architectural priorities. It categorizes technical debt into documentation, code quality, architecture, and security, providing a visible, prioritized strategy for community contribution.

## Status Tracking
- [ ] `Pending`
- [x] `Completed`

---

## Security

**Status: [ ] Pending | Priority: High**
- [ ] **Dependency Upgrades:** Upgrade outdated `gitpython` dependency (version 3.0.6) which has 5 known vulnerabilities (PYSEC-2022-42992, PYSEC-2023-137, PYSEC-2023-161, PYSEC-2023-165, and PYSEC-2024-4). This is a high-risk security item.

---

## Architecture

**Status: [x] Completed | Priority: High**
- [x] **Web Framework Consolidation:** Resolve web framework redundancy by consolidating the tech stack. The web framework has been consolidated to use Streamlit (`clintrials/visualization/dashboard/main.py`).

---

## Code Quality

**Status: [ ] Pending | Priority: Medium**
- [ ] **Core Logic Refactoring:** Refactor and rename the high-complexity `QuadrilateralRecruitmentStream` class in `clintrials/core/recruitment.py`.

**Status: [ ] Pending | Priority: Medium**
- [ ] **Numerical Instability / Anti-Patterns:** Vectorize the `posterior` method in `clintrials/dosefinding/crm.py` using `np.log` and `np.sum` instead of calculating likelihoods inefficiently using a loop and `np.prod`.

**Status: [ ] Pending | Priority: Low**
- [ ] **DRY Violations:** Extract repeated hardcoded values like `np.linspace(-5, 5, 1000)` in `clintrials/dosefinding/crm.py` into shared constants or variables.

**Status: [ ] Pending | Priority: Low**
- [ ] **Type Hinting & Error Handling:** Standardize type hints, docstrings, and explicit error handling codebase-wide.

---

## Documentation

**Status: [ ] Pending | Priority: Medium**
- [ ] **Maintenance Guidelines:** Expand contributing guidelines and standardize maintenance documentation, detailing procedures for handling recruitment and dose-finding logic transitions.
