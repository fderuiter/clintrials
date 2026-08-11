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
