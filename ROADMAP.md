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
