import os
import subprocess
import sys
import tempfile
from typing import Iterator, List

import pytest

# Add scripts directory to path to import audit_timeline
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts')))
from audit_timeline import audit_commits, get_module_name, is_test_file_for_module


def test_get_module_name() -> None:
    assert get_module_name('clintrials/core/recruitment.py') == 'recruitment'
    assert get_module_name('clintrials/core/__init__.py') is None
    assert get_module_name('other/core/recruitment.py') is None
    assert get_module_name('clintrials/core/data.txt') is None

def test_is_test_file_for_module() -> None:
    assert is_test_file_for_module('tests/test_recruitment.py', 'recruitment') is True
    assert is_test_file_for_module('tests/test_recruitment_utils.py', 'recruitment') is True
    assert is_test_file_for_module('tests/test_crm.py', 'recruitment') is False
    assert is_test_file_for_module('src/test_recruitment.py', 'recruitment') is False

import typing

if typing.TYPE_CHECKING:
    _F = typing.TypeVar('_F', bound=typing.Callable[..., typing.Any])
    def _fixture(func: _F) -> _F: return func
else:
    _fixture = pytest.fixture

@_fixture
def temp_git_repo(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    monkeypatch.delenv("GITHUB_HEAD_REF", raising=False)
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        try:
            # Initialize git repo
            subprocess.run(['git', 'init'], check=True, capture_output=True)
            subprocess.run(['git', 'config', 'user.email', 'test@example.com'], check=True)
            subprocess.run(['git', 'config', 'user.name', 'Test User'], check=True)
            subprocess.run(['git', 'config', 'commit.gpgsign', 'false'], check=True)

            # Create initial commit
            with open('README.md', 'w') as f:
                f.write('hello')
            subprocess.run(['git', 'add', 'README.md'], check=True)
            subprocess.run(['git', 'commit', '-m', 'Initial commit'], check=True)

            subprocess.run(['git', 'branch', '-m', 'main'], check=True)
            yield temp_dir
        finally:
            os.chdir(original_cwd)

def run_git(args: List[str]) -> None:
    subprocess.run(['git'] + args, check=True, capture_output=True)

def test_audit_commits_success(temp_git_repo: str) -> None:
    run_git(['checkout', '-b', 'feature-branch'])

    # Commit test first
    os.makedirs('tests', exist_ok=True)
    with open('tests/test_recruitment.py', 'w') as f:
        f.write('def test_foo(): pass')
    run_git(['add', 'tests/test_recruitment.py'])
    run_git(['commit', '-m', 'Add test'])

    # Commit implementation
    os.makedirs('clintrials/core', exist_ok=True)
    with open('clintrials/core/recruitment.py', 'w') as f:
        f.write('def foo(): pass')
    run_git(['add', 'clintrials/core/recruitment.py'])
    run_git(['commit', '-m', 'Add impl'])

    assert audit_commits('main', 'HEAD') is True

def test_audit_commits_failure(temp_git_repo: str) -> None:
    run_git(['checkout', '-b', 'feature-branch'])

    # Commit implementation without test
    os.makedirs('clintrials/core', exist_ok=True)
    with open('clintrials/core/recruitment.py', 'w') as f:
        f.write('def foo(): pass')
    run_git(['add', 'clintrials/core/recruitment.py'])
    run_git(['commit', '-m', 'Add impl'])

    assert audit_commits('main', 'HEAD') is False

def test_audit_commits_concurrent(temp_git_repo: str) -> None:
    run_git(['checkout', '-b', 'feature-branch'])

    # Commit both at the same time
    os.makedirs('tests', exist_ok=True)
    with open('tests/test_recruitment.py', 'w') as f:
        f.write('def test_foo(): pass')

    os.makedirs('clintrials/core', exist_ok=True)
    with open('clintrials/core/recruitment.py', 'w') as f:
        f.write('def foo(): pass')

    run_git(['add', 'tests/test_recruitment.py', 'clintrials/core/recruitment.py'])
    run_git(['commit', '-m', 'Add both'])

    assert audit_commits('main', 'HEAD') is True

def test_audit_commits_hotfix(temp_git_repo: str) -> None:
    run_git(['checkout', '-b', 'hotfix/urgent-fix'])

    # Commit implementation without test
    os.makedirs('clintrials/core', exist_ok=True)
    with open('clintrials/core/recruitment.py', 'w') as f:
        f.write('def foo(): pass')
    run_git(['add', 'clintrials/core/recruitment.py'])
    run_git(['commit', '-m', 'Add impl'])

    # Should pass because of branch name
    assert audit_commits('main', 'HEAD') is True

def test_audit_commits_hotfix_override(temp_git_repo: str) -> None:
    run_git(['checkout', '-b', 'feature-branch'])

    # Commit implementation without test
    os.makedirs('clintrials/core', exist_ok=True)
    with open('clintrials/core/recruitment.py', 'w') as f:
        f.write('def foo(): pass')
    run_git(['add', 'clintrials/core/recruitment.py'])
    run_git(['commit', '-m', 'Add impl'])

    # Should pass because of override branch name
    assert audit_commits('main', 'HEAD', override_branch_name='hotfix/override') is True

def test_audit_commits_skip_tdd_trailer(temp_git_repo: str) -> None:
    run_git(['checkout', '-b', 'feature-branch'])

    # Commit implementation without test, but with skip-tdd
    os.makedirs('clintrials/core', exist_ok=True)
    with open('clintrials/core/recruitment.py', 'w') as f:
        f.write('def foo(): pass')
    run_git(['add', 'clintrials/core/recruitment.py'])
    run_git(['commit', '-m', 'Add impl\n\nskip-tdd: true'])

    assert audit_commits('main', 'HEAD') is True

def test_audit_commits_existing_test(temp_git_repo):
    # Commit test on main branch first
    os.makedirs('tests', exist_ok=True)
    with open('tests/test_recruitment.py', 'w') as f:
        f.write('def test_foo(): pass')
    
    os.makedirs('clintrials/core', exist_ok=True)
    with open('clintrials/core/recruitment.py', 'w') as f:
        f.write('def foo(): pass')
        
    run_git(['add', 'tests/test_recruitment.py', 'clintrials/core/recruitment.py'])
    run_git(['commit', '-m', 'Add test and impl on main'])
    
    # Create feature branch and modify only the implementation
    run_git(['checkout', '-b', 'feature-branch'])
    with open('clintrials/core/recruitment.py', 'w') as f:
        f.write('def foo(): return 42')
    run_git(['add', 'clintrials/core/recruitment.py'])
    run_git(['commit', '-m', 'Modify impl'])
    
    # Audit should pass because test exists in tree
    assert audit_commits('main', 'HEAD') is True
