# SPDX-License-Identifier: MIT

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from clintrials.validation import validate_feature_request


def test_missing_personas_json():
    original_exists = Path.exists
    def side_effect_exists(self):
        if "personas.json" in str(self):
            return False
        return original_exists(self)

    with patch.object(Path, "exists", autospec=True, side_effect=side_effect_exists):
        with pytest.raises(ValueError, match="External JSON configuration file is missing: personas.json"):
            validate_feature_request({"track": "user-centric"})


def test_empty_personas_json():
    original_exists = Path.exists
    def side_effect_exists(self):
        if "personas.json" in str(self):
            return True
        return original_exists(self)

    original_read_text = Path.read_text
    def side_effect_read_text(self):
        if "personas.json" in str(self):
            return ""
        return original_read_text(self)

    with patch.object(Path, "exists", autospec=True, side_effect=side_effect_exists), \
         patch.object(Path, "read_text", autospec=True, side_effect=side_effect_read_text):
        with pytest.raises(ValueError, match="External JSON configuration file is empty: personas.json"):
            validate_feature_request({"track": "user-centric"})


def test_invalid_json_personas():
    original_exists = Path.exists
    def side_effect_exists(self):
        if "personas.json" in str(self):
            return True
        return original_exists(self)

    original_read_text = Path.read_text
    def side_effect_read_text(self):
        if "personas.json" in str(self):
            return "{malformed json"
        return original_read_text(self)

    with patch.object(Path, "exists", autospec=True, side_effect=side_effect_exists), \
         patch.object(Path, "read_text", autospec=True, side_effect=side_effect_read_text):
        with pytest.raises(ValueError, match="External JSON configuration file is structurally invalid"):
            validate_feature_request({"track": "user-centric"})


def test_invalid_structure_personas():
    original_exists = Path.exists
    def side_effect_exists(self):
        if "personas.json" in str(self):
            return True
        return original_exists(self)

    original_read_text = Path.read_text
    def side_effect_read_text(self):
        if "personas.json" in str(self):
            return '{"name": "some persona"}'
        return original_read_text(self)

    with patch.object(Path, "exists", autospec=True, side_effect=side_effect_exists), \
         patch.object(Path, "read_text", autospec=True, side_effect=side_effect_read_text):
        with pytest.raises(ValueError, match="External JSON configuration must be a list of persona definitions."):
            validate_feature_request({"track": "user-centric"})


def test_invalid_item_personas():
    original_exists = Path.exists
    def side_effect_exists(self):
        if "personas.json" in str(self):
            return True
        return original_exists(self)

    original_read_text = Path.read_text
    def side_effect_read_text(self):
        if "personas.json" in str(self):
            return '[{"no_name": "some persona"}]'
        return original_read_text(self)

    with patch.object(Path, "exists", autospec=True, side_effect=side_effect_exists), \
         patch.object(Path, "read_text", autospec=True, side_effect=side_effect_read_text):
        with pytest.raises(ValueError, match="Each persona definition must be a JSON object with a 'name' field."):
            validate_feature_request({"track": "user-centric"})


def test_custom_dynamic_personas_and_aliases():
    custom_config = [
        {
            "name": "visionary leader",
            "aliases": ["visionary leader", "steve jobs", "visionary"]
        }
    ]

    original_exists = Path.exists
    def side_effect_exists(self):
        if "personas.json" in str(self) or "PRODUCT_STRATEGY.md" in str(self):
            return True
        return original_exists(self)

    original_read_text = Path.read_text
    def side_effect_read_text(self):
        if "personas.json" in str(self):
            return json.dumps(custom_config)
        if "PRODUCT_STRATEGY.md" in str(self):
            return "Visionary Leader is a key strategic persona."
        return original_read_text(self)

    with patch.object(Path, "exists", autospec=True, side_effect=side_effect_exists), \
         patch.object(Path, "read_text", autospec=True, side_effect=side_effect_read_text):

        # Test valid match of custom persona
        issue_data_valid = {
            "track": "user-centric",
            "persona": "As a visionary, I want to disrupt the clinical trial industry."
        }
        assert validate_feature_request(issue_data_valid) is True

        # Test invalid/unrecognized persona check
        issue_data_invalid = {
            "track": "user-centric",
            "persona": "Some regular employee wants a coffee machine."
        }
        with pytest.raises(ValueError, match="User-Centric / Clinical track proposals must reference a defined strategic persona"):
            validate_feature_request(issue_data_invalid)
