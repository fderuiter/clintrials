"""Dashboard CLI launcher."""
from __future__ import annotations
# ruff: noqa: T201
import argparse
import os
import sys


def main():  # type: ignore
    """Launch the Clinical Trials Dashboard Streamlit app programmatically."""
    parser = argparse.ArgumentParser(description="Programmatic CLI Launcher for Clinical Trials Dashboard")
    parser.add_argument("--host", "-H", type=str, default=None, help="The host address to bind the server to.")
    parser.add_argument("--port", "-p", type=int, default=None, help="The port to run the server on.")
    # Support unknown args so standard Streamlit args can also be forwarded
    args, unknown = parser.parse_known_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    main_script_path = os.path.join(current_dir, "main.py")

    print("Starting Clinical Trials Dashboard...")
    print(f"Main script: {main_script_path}")

    streamlit_args = ["streamlit", "run", main_script_path]
    if args.host:
        streamlit_args.extend(["--server.address", args.host])
        print(f"Host configured to: {args.host}")
    if args.port:
        streamlit_args.extend(["--server.port", str(args.port)])
        print(f"Port configured to: {args.port}")

    sys.argv = streamlit_args + unknown

    print("Launching Streamlit server programmatically...")
    try:
        try:
            import streamlit.web.cli as stcli
        except ImportError:
            import streamlit.cli as stcli

        stcli.main()
    except KeyboardInterrupt:
        print("\nStopping Streamlit server... Safe shutdown initiated.")
        sys.exit(0)

if __name__ == "__main__":
    main()  # type: ignore
