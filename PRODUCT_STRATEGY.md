# Product Strategy & Validation Framework

This document outlines the strategic framing, personas, target environments, and validation frameworks for `clintrials`. It acts as the strategic product pillar to ensure that software development respects safety requirements, regulatory alignment, intuitive interfaces for clinical stakeholders, and robust programmatic tools for developers and biostatisticians.

---

## 1. User Personas

To guide the development of user experiences and safeguards, we define the following primary personas:

### Clinical Persona A: Dr. Aris Thorne (Principal Investigator / Clinician)
- **Role:** Leads clinical trial execution at active medical centers.
- **Goal:** Wants to ensure patient safety remains paramount while utilizing adaptive designs.
- **Needs:**
  - Clear explanations of algorithm-recommended doses.
  - The absolute ability to perform **manual clinical overrides** when a patient's individual clinical profile (e.g., unexpected comorbidities or non-DLT toxicities) makes the algorithmically recommended dose inappropriate.
  - Transparent displays of safety margins.
- **Pain Point:** Systems that operate as a "black box" or enforce rigid recommendations without allowing clinical overridden paths.

### Clinical Persona B: Eleanor Vance (Clinical Trial Manager)
- **Role:** Oversees multi-center coordination, safety committee reviews, and protocol compliance.
- **Goal:** Verify that static dose transition pathways and cumulative safety limits are enforced.
- **Needs:**
  - Automated verification of dose escalation pathways.
  - Clear visualizations of simulated trials to present during Safety Cohort Review meetings.
  - Reliable Streamlit dashboards that initialize instantly with correct parameters.
- **Pain Point:** UI crashes during presentation, or hidden/nested dose pathways that cannot be cross-referenced with the protocol documentation.

### Programmatic Persona C: Biostatistician (Trial Designer & Simulation Engineer)
- **Role:** Designs trial simulations and evaluates operating characteristics of complex adaptive protocols.
- **Goal:** Submit proposals and build custom statistical simulation endpoints without clinical persona constraints.
- **Needs:**
  - Flexible programmatic APIs to specify priors, models, and custom clinical simulation scenarios.
  - Efficient vectorized math routines and numerical stability guarantees.
- **Pain Point:** Being forced to shoehorn programmatic simulation requirements into irrelevant clinical personas during triage.

### Programmatic Persona D: Data Scientist / Developer (Core Infrastructure Engineer)
- **Role:** Implements core machine learning/statistical libraries and maintains system infrastructure.
- **Goal:** Improve code health, refactor core modules, fix class complexity, and update type annotations.
- **Needs:**
  - Rigid static typing and robust CI checks.
  - Well-modularized code to avoid monolithic class designs.
- **Pain Point:** Technical debt and high class complexity in modules like recruitment causing integration friction.

---

## 2. Target Environments

Our software must operate reliably across diverse clinical and programmatic environments:
1. **Interactive Safety Review (Streamlit Dashboard):** Used during safety committee reviews to visualize dynamic/static transition pathways. Must load robustly without crash.
2. **Pre-trial Simulation (Dry Runs):** Run by biostatisticians and trial designers to evaluate operating characteristics (e.g., probability of selecting the optimal dose, expected number of patients treated at each dose) under different true toxicity/efficacy scenarios.
3. **In-trial Monitoring:** Tracking real-time patient cohorts and computing the next dose recommendation under active protocol rules.

---

## 3. Product Validation Framework

To prevent late-stage, expensive redesigns of functions, we implement a **Dual-Track Verification Workflow** where user-centric concerns are validated programmatically alongside infrastructure benchmarks.

```
                  [ Feature / Issue Intake ]
                              |
                +-------------+-------------+
                |                           |
     [ Infrastructure Track ]        [ User-Centric Track ]
                |                           |
     (Milestones in ROADMAP.md)   (Pillars in PRODUCT_STRATEGY.md)
                |                           |
                +-------------+-------------+
                              |
                     [ Automated Tests ]
                  - Persona overrides
                  - Static dose transitions
                  - Headless UI validation
```

### Validation Pillars:
1. **Clinician Override Verification:** Every adaptive design implementation must support a manual override mechanism. Tests must simulate a clinician overriding dynamic recommendations (e.g., CRM or EffTox recommending Dose 3, but the clinician overriding to Dose 2) and verify that subsequent cohort actions utilize this override correctly.
2. **Static Dose Transition Verification:** Transition logic must be validated against expected mathematical outputs for standard safety models.
3. **Headless Visual Validation:** Dashboard components must be tested automatically in a headless test environment to ensure parameter loading is crash-free.

---

## 4. Software Qualification & GxP Compliance Framework

To satisfy regulatory standards (such as FDA 21 CFR Part 11, Annex 11, and Good Clinical Practice guidelines), `clintrials` integrates a comprehensive Software Qualification framework. This framework is implemented within the library's design to verify that installation, operations, and performance consistently meet quality and accuracy benchmarks.

### A. Installation, Operational, and Performance Qualification (IQ/OQ/PQ)

The software qualification lifecycle is formalized into three operational phases, supported directly by the codebase's verification tools:

1. **Installation Qualification (IQ):**
   * **Objective:** Verify that the software is installed correctly, with all expected packages, dependencies, and configuration parameters in place.
   * **Verification Process:** The qualification CLI collects system-level details (OS, Python version, dependency libraries like NumPy, Pandas, SciPy, Statsmodels, and fpdf2). It computes a recursive **SHA-256 cryptographic signature** of all source files in the `clintrials` package. This signature acts as a tamper-evident seal, ensuring the deployed codebase matches the certified release and has not been altered post-installation.

2. **Operational Qualification (OQ):**
   * **Objective:** Ensure that all core statistical calculations, decision algorithms, and adaptive dose-finding components function exactly as specified.
   * **Verification Process:** The CLI runs a deterministic suite of 31 operational tests, assessing mathematical outputs under controlled scenarios against strict tolerance thresholds (e.g., probability estimates and boundary calculations).

3. **Performance Qualification (PQ):**
   * **Objective:** Ensure that the software maintains high precision, stability, and absolute reproducibility under realistic, concurrent clinical simulation workloads.
   * **Verification Process:** Handled via decoupled, stateless execution paths that leverage thread-safe random number generation. This ensures that concurrent simulation threads do not leak random state, producing identical operational characteristics across high-throughput simulation runs.

---

### B. Isolated Random Number Generation & Simulation Reproducibility

Computational reproducibility is a core non-negotiable requirement for clinical study setup and simulation-driven trial design. Traditional global pseudo-random number generator (PRNG) states are prone to cross-thread/process pollution and side effects, especially in parallel execution or web-dashboard contexts.

To enforce absolute reproducibility:
* **Decoupled Generator Objects:** `clintrials` uses a centralized utility `clintrials.core.rng.get_rng(seed)` which returns an independent, isolated instance of `numpy.random.Generator` (created using PCG64 via `numpy.random.default_rng(seed)`).
* **Stateless Simulation Paths:** All simulation functions must accept an explicit `rng` object or a specific `seed`. Random operations are performed strictly using this isolated generator instance.
* **Concurrency Protection:** Since global states are never read or modified during execution, concurrent/parallel simulation paths are guaranteed to be side-effect-free, producing identical statistical characteristics regardless of parallel orchestration or UI interactions.

---

### C. Reference Statistical Baselines

The GxP validation CLI compares the library's outputs against verified, peer-reviewed textbook and R-package baselines:

1. **Bayesian CRM (Continual Reassessment Method):**
   * **Baseline Source:** Ken Cheung (2011), *Clinical Trial Design Using Cumulative Cohort Design*, Table 3.2 reference dataset.
   * **Comparison Metrics:** Verifies the posterior model parameter estimate ($\hat{\beta}$) at sequential patient milestones, matching Cheung's reference values within an absolute tolerance of $\epsilon \le 0.005$.

2. **MLE CRM (Maximum Likelihood Estimation):**
   * **Baseline Source:** The industry-standard R package `dfcrm` (Dose-Finding CRM, developed by Ken Cheung).
   * **Comparison Metrics:** Evaluates $\hat{\beta}$ calculations at consecutive milestone checkpoints against `dfcrm` results under an absolute tolerance of $\epsilon \le 0.005$.

3. **EffTox (Efficacy-Toxicity Bayesian Adaptive Design):**
   * **Baseline Source:** Thall, Cook, and Estey (2014) Cohort 1 published reference dataset.
   * **Comparison Metrics:** Validates the next recommended dose level (exact match) and probability of efficacy ($\text{Pr}_{\text{eff}}$) and probability of toxicity ($\text{Pr}_{\text{tox}}$) at each dose level within an absolute tolerance of $\epsilon \le 0.08$.

4. **Group Sequential Design (GSD):**
   * **Baseline Source:** Standard statistical textbook boundaries for group sequential trials using O'Brien-Fleming spending functions.
   * **Comparison Metrics:** Confirms that efficacy boundaries generated at each of the 4 design stages match the reference boundaries within a relative tolerance of $\text{rtol} \le 0.08$.

---

### D. Accessible PDF Structural Compliance

Regulatory audits require documentation to be universally accessible, complying with Section 508 and WCAG/PDF-UA standards to ensure compatibility with screen readers. The generated qualification report PDF is engineered for strict accessibility conformance:

* **Document Metadata & Language:** Establishes the standard PDF version 1.7, assigns the explicit document language (`en-US`), and enables outline/document-title display settings.
* **Semantic Tagging Hierarchy:** Automatically generates structured PDF tags mapping content elements to logical entities:
  * `/H1` tags for headings.
  * `/P` tags for narrative paragraphs.
  * `/Table` container tags containing `/TR` (table rows), `/TH` (table header cells), and `/TD` (table data cells).
* **Leaf-Level MCID Assignment:** Complies with PDF/UA structural rules by only assigning Marked Content Identifiers (`/MCID`) to structural leaves (such as text fragments inside `/TD`), keeping container structures like `/Table` or `/TR` purely structural without direct MCIDs.
* **Decorative Artifacting:** Utilizes `/Artifact` blocks to encapsulate horizontal rules and purely visual styling decorators, ensuring screen readers bypass non-narrative components.

