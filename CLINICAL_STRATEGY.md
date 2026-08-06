# Clinical Strategy & Validation Framework

This document outlines the clinical framing, personas, target trial environments, and validation frameworks for `clintrials`. It acts as the strategic clinical pillar to ensure that software development respects safety requirements, regulatory alignment, and intuitive interfaces for non-technical clinical stakeholders.

---

## 1. Clinical Personas

To guide the development of user experiences and safety safeguards, we define two primary clinical personas:

### Persona A: Dr. Aris Thorne (Principal Investigator / Clinician)
- **Role:** Leads clinical trial execution at active medical centers.
- **Goal:** Wants to ensure patient safety remains paramount while utilizing adaptive designs.
- **Needs:**
  - Clear explanations of algorithm-recommended doses.
  - The absolute ability to perform **manual clinical overrides** when a patient's individual clinical profile (e.g., unexpected comorbidities or non-DLT toxicities) makes the algorithmically recommended dose inappropriate.
  - Transparent displays of safety margins.
- **Pain Point:** Systems that operate as a "black box" or enforce rigid recommendations without allowing clinical overridden paths.

### Persona B: Eleanor Vance (Clinical Trial Manager)
- **Role:** Oversees multi-center coordination, safety committee reviews, and protocol compliance.
- **Goal:** Verify that static dose transition pathways and cumulative safety limits are enforced.
- **Needs:**
  - Automated verification of dose escalation pathways.
  - Clear visualizations of simulated trials to present during Safety Cohort Review meetings.
  - Reliable Streamlit dashboards that initialize instantly with correct parameters.
- **Pain Point:** UI crashes during presentation, or hidden/nested dose pathways that cannot be cross-referenced with the protocol documentation.

---

## 2. Target Trial Environments

Our software must operate reliably across diverse clinical environments:
1. **Interactive Safety Review (Streamlit Dashboard):** Used during safety committee reviews to visualize dynamic/static transition pathways. Must load robustly without crash.
2. **Pre-trial Simulation (Dry Runs):** Run by trial designers to evaluate operating characteristics (e.g., probability of selecting the optimal dose, expected number of patients treated at each dose) under different true toxicity/efficacy scenarios.
3. **In-trial Monitoring:** Tracking real-time patient cohorts and computing the next dose recommendation under active protocol rules.

---

## 3. Clinical Validation Framework

To prevent late-stage, expensive redesigns of safety functions, we implement a **Dual-Track Verification Workflow** where clinical concerns are validated programmatically alongside technical benchmarks.

```
                  [ Feature / Issue Intake ]
                              |
                +-------------+-------------+
                |                           |
     [ Technical Track ]           [ Clinical Track ]
                |                           |
     (Milestones in ROADMAP.md)   (Pillars in CLINICAL_STRATEGY.md)
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
