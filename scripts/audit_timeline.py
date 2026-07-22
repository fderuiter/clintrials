#!/usr/bin/env python3
"""Script to audit Git commit timelines for TDD compliance."""
import os
import re
import subprocess
import sys


def run_git(args):
    """Execute a git command and return its output."""
    result = subprocess.run(['git'] + args, capture_output=True, text=True)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, ['git'] + args, result.stdout, result.stderr)
    return result.stdout.strip()

def get_module_name(file_path):
    """Extract the module name from the source file path.

    For example, clintrials/core/recruitment.py -> recruitment.
    """
    if not file_path.startswith('clintrials/') or not file_path.endswith('.py'):
        return None
    if file_path.endswith('__init__.py'):
        return None
    basename = os.path.basename(file_path)
    return basename[:-3]

def is_test_file_for_module(test_path, module_name):
    """Check if a given test file corresponds to the module."""
    if not test_path.startswith('tests/') or not test_path.endswith('.py'):
        return False
    filename = os.path.basename(test_path)[:-3]
    if filename == f"test_{module_name}" or filename.startswith(f"test_{module_name}_"):
        return True
    return False

def audit_commits(base_ref, head_ref, override_branch_name=None):
    """Audit commits to ensure tests are written before or alongside implementation."""
    # 1. Check if branch starts with hotfix/
    branch_name = override_branch_name or os.environ.get('GITHUB_HEAD_REF')
    if not branch_name:
        try:
            branch_name = run_git(['rev-parse', '--abbrev-ref', 'HEAD'])
        except Exception:
            branch_name = ""

    if branch_name and branch_name.startswith('hotfix/'):
        print(f"Skipping TDD audit for hotfix branch: {branch_name}")  # noqa: T201
        return True

    # 2. Get list of commits in the PR
    try:
        commits_output = run_git(['log', f'{base_ref}..{head_ref}', '--reverse', '--format=%H'])
    except subprocess.CalledProcessError as e:
        print(f"Error getting commits: {e.stderr}")  # noqa: T201
        return False

    commits = [c for c in commits_output.split('\n') if c.strip()]
    if not commits:
        print("No commits found in PR range.")  # noqa: T201
        return True

    seen_tests = set()

    for commit in commits:
        # 3. Check for skip-tdd trailer in this commit message
        msg = run_git(['log', '-1', '--format=%B', commit])
        if re.search(r'(?im)^skip-tdd\b', msg):
            print(f"Skipping TDD audit due to skip-tdd trailer in commit {commit}.")  # noqa: T201
            return True

        # Check modified files in this commit
        files_output = run_git(['show', '--name-only', '--format=', commit])
        files = [f for f in files_output.split('\n') if f.strip()]

        # Add tests to seen_tests (before checking source files in this commit,
        # as tests could be modified in the same commit)
        for f in files:
            if f.startswith('tests/') and f.endswith('.py'):
                seen_tests.add(f)

        # Now check source files
        for f in files:
            module_name = get_module_name(f)
            if module_name:
                test_found = any(is_test_file_for_module(t, module_name) for t in seen_tests)
                if not test_found:
                    print(f"TDD Audit Failed: Commit {commit} modifies source file '{f}' (module: '{module_name}') "  # noqa: T201
                          f"but no corresponding test file (e.g. 'tests/test_{module_name}.py') was added/modified "
                          "in this or any preceding commit.")
                    return False

    print("TDD Audit passed.")  # noqa: T201
    return True

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: audit_timeline.py <base_ref> <head_ref>")  # noqa: T201
        sys.exit(1)

    base = sys.argv[1]
    head = sys.argv[2]

    if audit_commits(base, head):
        sys.exit(0)
    else:
        sys.exit(1)
