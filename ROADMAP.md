# Roadmap

This document outlines the strategic priorities for the project, focusing on technical milestones, security improvements, and structural technical debt.

## 1. Milestones
- Modularize core numerical integration and probability models for better reusability.
- Stabilize the public API for phase 3 sequential designs and ensure complete type annotation coverage.
- Improve test coverage for the Trial Simulation Hub and expand available visualizations.

## 2. Security Priorities
- Proactively monitor and resolve vulnerabilities in high-risk dependencies.
- Ensure strict compliance with standardized disclosure policies for identified issues.
- Continuously audit and update the continuous integration pipeline for secure execution.

## 3. Technical Debt
- Decompose complex monolith classes (e.g., in the recruitment module) into composable and maintainable units.
- Modernize mathematical operations in dose-finding algorithms by prioritizing stable, vectorized operations over iterative linear products.
- Enforce strict static type checking across the entire codebase to reduce integration failures and minimize debugging cycles.

## 4. GxP Compliance & Automation Timeline

To elevate trust and automate regulatory readiness, the project commits to the following milestones for continuous compliance validation and qualification artifact retention:

### Milestone 4.1: Continuous Compliance Integration (Q3 2026)
* **Objective:** Prevent mathematical regressions and ensure installation integrity is verified continuously.
* **Tasks:**
  * Integrate the GxP qualification CLI (`clintrials-validate --verbose`) as a required health check in the GitHub Actions continuous integration (CI) pipeline.
  * Reject pull requests that fail to satisfy mathematical precision boundaries ($\epsilon$ tolerances) against our standard reference baselines.
  * Automate verification that documentation files (`CLINICAL_STRATEGY.md`) remain synchronized with changes in reference statistics.

### Milestone 4.2: Automated Qualification Archiving (Q4 2026)
* **Objective:** Guarantee immutable, traceable verification history for institutional buyers and auditors.
* **Tasks:**
  * Establish an automated release-tagging hook to compile the GxP qualification PDF (`clintrials_validation_report.pdf`) upon every official library tag.
  * Automatically archive signed and generated PDFs in safe, audit-ready storage (e.g., as immutable GitHub Release Assets or designated regulatory repository archives).
  * Introduce deterministic configuration logging inside the qualification runner to seal each report with the specific git commit hash and dependency dependency versions.

