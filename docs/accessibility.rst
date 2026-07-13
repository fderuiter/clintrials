Accessibility Guidelines
========================

Unified Layout Design
---------------------

In our continued effort to make clinical trial data accessible to everyone without barriers, we have adopted a **Unified Inline Layout** strategy for the Streamlit dashboard.

Key Design Principles:
1. **Permanent Inline Summaries**: Data summaries are permanently rendered in plain text below each chart. No interactive collapse widgets (expanders) or hidden tabs are used. This ensures screen readers announce the critical data during standard sweeps.
2. **Zero Configuration**: Users are not required to locate or configure an "Accessibility Mode" toggle. High-fidelity semantic text and clear contrast are default behaviors across the entire interface.
3. **Responsive Dimensioning**: The unified layout retains the default chart dimensions and standard spacing, providing an optimized visual flow for standard viewports without UI overlap.

Accessibility Testing Guidelines
--------------------------------

We maintain strict accessibility compliance through automated WCAG audits running via Playwright and the Axe engine.

1. **Comprehensive Scope**: The audits run against the fully rendered page. No exclusions or ignored elements are permitted, meaning sidebars, interactive controls, and native framework components are strictly checked.
2. **Zero Tolerance Violations**: Tests will automatically fail the build if ANY WCAG violation is detected. This encompasses all severity levels: critical, serious, moderate, and minor impacts.
3. **Dynamic Landmark Roles**: The application dynamically injects ARIA landmarks (`role="main"`, `role="banner"`, `role="navigation"`) to wrapper structures and corrects invalid properties (such as unauthorized `aria-expanded` attributes on non-expandable sidebars) so that semantic structure conforms to standard requirements. 

Running the Tests Locally
-------------------------

Developers should verify their changes by running the automated accessibility suite:

.. code-block:: bash

   poetry run playwright install
   poetry run pytest tests/test_accessibility.py

If any violations are reported, you must correct the component rendering or adjust the DOM structural injections in the main layout configuration before committing code.
