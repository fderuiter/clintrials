# SPDX-License-Identifier: MIT

import pytest

from clintrials.validation import validate_feature_request


def test_valid_clinical_track_feature_request():
    issue_data = {
        "track": "User-Centric / Clinical Track",
        "clinical_pillar": "Persona A: Dr. Aris Thorne - Manual clinician override capability",
        "solution_description": "Add override button to simulation view for safety overrides.",
    }
    # Should validate against PRODUCT_STRATEGY.md and succeed
    assert validate_feature_request(issue_data) is True


def test_invalid_clinical_track_feature_request():
    issue_data = {
        "track": "User-Centric / Clinical Track",
        "clinical_pillar": "",  # Missing
    }
    with pytest.raises(
        ValueError, match="User-Centric / Clinical track requires a reference to PRODUCT_STRATEGY.md"
    ):
        validate_feature_request(issue_data)


def test_valid_technical_track_feature_request():
    issue_data = {
        "track": "Technical / Infrastructure Track",
        "roadmap_milestone": "Modularize core numerical integration and probability models",
        "solution_description": "Optimize posterior calculations in CRM model.",
    }
    assert validate_feature_request(issue_data) is True


def test_invalid_technical_track_feature_request():
    issue_data = {"track": "Technical / Infrastructure Track", "roadmap_milestone": "None"}
    with pytest.raises(
        ValueError,
        match="Technical / Infrastructure track requires a valid ROADMAP.md milestone reference",
    ):
        validate_feature_request(issue_data)


def test_invalid_track_selection():
    issue_data = {"track": "unknown_track", "roadmap_milestone": "Some milestone"}
    with pytest.raises(ValueError, match="Invalid track selection"):
        validate_feature_request(issue_data)


def test_valid_user_centric_track_feature_request_clinical():
    issue_data = {
        "track": "User-Centric / Clinical Track",
        "user_persona": "Dr. Aris Thorne needs custom statistical override capability",
        "solution_description": "Add override button to simulation view.",
    }
    assert validate_feature_request(issue_data) is True


def test_valid_user_centric_track_feature_request_programmatic_biostatistician():
    issue_data = {
        "track": "User-Centric / Clinical Track",
        "persona": "As a biostatistician, I want custom statistical simulation endpoints.",
        "solution_description": "Implement custom simulation endpoints.",
    }
    assert validate_feature_request(issue_data) is True


def test_valid_user_centric_track_feature_request_programmatic_data_scientist():
    issue_data = {
        "track": "User-Centric / Clinical Track",
        "clinical_pillar": "As a data scientist, I need robust vectorized operations.",
        "solution_description": "Vectorize CRM calculations.",
    }
    assert validate_feature_request(issue_data) is True


def test_invalid_user_centric_track_no_persona():
    issue_data = {
        "track": "User-Centric / Clinical Track",
        "persona": "",
    }
    with pytest.raises(ValueError, match="User-Centric / Clinical track requires a reference to PRODUCT_STRATEGY.md"):
        validate_feature_request(issue_data)


def test_invalid_user_centric_track_unrecognized_persona():
    issue_data = {
        "track": "User-Centric / Clinical Track",
        "persona": "An external partner wants better performance.",
    }
    with pytest.raises(ValueError, match="User-Centric / Clinical track proposals must reference a defined strategic persona"):
        validate_feature_request(issue_data)


def test_valid_infrastructure_track_feature_request():
    issue_data = {
        "track": "Technical / Infrastructure Track",
        "roadmap_milestone": "Modularize core numerical integration and probability models",
        "solution_description": "Refactor integration module.",
    }
    assert validate_feature_request(issue_data) is True


def test_invalid_infrastructure_track_no_milestone():
    issue_data = {
        "track": "Technical / Infrastructure Track",
        "roadmap_milestone": "",
    }
    with pytest.raises(ValueError, match="Technical / Infrastructure track requires a valid ROADMAP.md milestone reference"):
        validate_feature_request(issue_data)


def test_invalid_infrastructure_track_unrecognized_milestone():
    issue_data = {
        "track": "Technical / Infrastructure Track",
        "roadmap_milestone": "Build an artificial intelligence chatbot.",
    }
    with pytest.raises(ValueError, match="Technical / Infrastructure track requires a valid and active ROADMAP.md milestone reference"):
        validate_feature_request(issue_data)
