# SPDX-License-Identifier: MIT

import pytest

from clintrials.validation import validate_feature_request


def test_valid_clinical_track_feature_request():
    issue_data = {
        "track": "Clinical Track (Bypasses ROADMAP.md; aligns with CLINICAL_STRATEGY.md)",
        "clinical_pillar": "Persona A: Dr. Aris Thorne - Manual clinician override capability",
        "solution_description": "Add override button to simulation view for safety overrides."
    }
    # Should bypass technical roadmap milestone and succeed
    assert validate_feature_request(issue_data) is True


def test_invalid_clinical_track_feature_request():
    issue_data = {
        "track": "clinical",
        "clinical_pillar": ""  # Missing
    }
    with pytest.raises(ValueError, match="Clinical track requires a reference to CLINICAL_STRATEGY.md"):
        validate_feature_request(issue_data)


def test_valid_technical_track_feature_request():
    issue_data = {
        "track": "technical",
        "roadmap_milestone": "Section 1: Technical milestones - CRM optimizations",
        "solution_description": "Optimize posterior calculations in CRM model."
    }
    assert validate_feature_request(issue_data) is True


def test_invalid_technical_track_feature_request():
    issue_data = {
        "track": "technical",
        "roadmap_milestone": "None"
    }
    with pytest.raises(ValueError, match="Technical track requires a valid ROADMAP.md milestone reference"):
        validate_feature_request(issue_data)


def test_invalid_track_selection():
    issue_data = {
        "track": "unknown_track",
        "roadmap_milestone": "Some milestone"
    }
    with pytest.raises(ValueError, match="Invalid track selection"):
        validate_feature_request(issue_data)
