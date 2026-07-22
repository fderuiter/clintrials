"""Utility functions for dashboard interactions and accessibility."""

from __future__ import annotations

import uuid

import streamlit as st


def announce_status_locally(message: str, key: str = None):  # type: ignore
    """Renders an invisible, sandboxed ARIA live region locally to announce dynamic updates.

    This is SOP/CSP compliant.
    """
    if not st.session_state.get("accessibility_mode", False):
        return

    try:
        import streamlit.components.v1 as components
    except Exception:
        # Failsafe for mocked test environments where submodule imports fail
        return

    if not key:
        key = f"sr-announce-{uuid.uuid4().hex[:8]}"

    # We use height=0 to occupy zero pixels and prevent layout shifting.
    # The container acts as a polite aria-live region.
    html_str = f"""
    <div id="sr-announcement" aria-live="polite" style="position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden; clip: rect(0, 0, 0, 0); border: 0;"></div>
    <script>
        setTimeout(() => {{
            const el = document.getElementById("sr-announcement");
            if (el) el.textContent = "{message}";
        }}, 100);
        setTimeout(() => {{
            const el = document.getElementById("sr-announcement");
            if (el) el.textContent = "";
        }}, 3000);
    </script>
    """
    components.html(html_str, height=0, key=key)
