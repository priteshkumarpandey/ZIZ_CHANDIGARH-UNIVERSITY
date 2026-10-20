import re
from pathlib import Path

import pytest


def check_no_unverified_license_claimed(text: str, filename: str) -> None:
    # Rule: The words 'verified' or 'confirmed' next to a license are allowed for CGMacros and HUPA-UCM only.
    # If any other source (AI-READI, Synthea, IFCT) is claimed with a verified or confirmed license, fail.
    unverified_sources = ["AI-READI", "Synthea", "IFCT"]
    license_names = ["MIT", "Apache", "BSD", "GPL", "Creative Commons", "CC BY", "CC-BY", "ODbL"]

    for line in text.splitlines():
        line_clean = line.strip()
        for src in unverified_sources:
            if src.lower() in line_clean.lower():
                for lic in license_names:
                    # Pattern matching a named license accompanied by 'verified' or 'confirmed' on the same line
                    pat = re.compile(
                        rf"\b{re.escape(lic)}\b.*?\b(verified|confirmed)\b|\b(verified|confirmed)\b.*?\b{re.escape(lic)}\b",
                        re.IGNORECASE,
                    )
                    if pat.search(line_clean):
                        raise ValueError(
                            f"File {filename} invalid: found 'verified' or 'confirmed' next to license '{lic}' for source '{src}': '{line_clean}'"
                        )


def test_notice_and_data_licenses_files_exist():
    notice_path = Path("models/checkpoints/NOTICE.md")
    data_licenses_path = Path("DATA_LICENSES.md")

    assert notice_path.exists(), "models/checkpoints/NOTICE.md is missing"
    assert data_licenses_path.exists(), "DATA_LICENSES.md is missing"

    notice_content = notice_path.read_text(encoding="utf-8")
    data_licenses_content = data_licenses_path.read_text(encoding="utf-8")

    assert len(notice_content) > 50
    assert len(data_licenses_content) > 100

    # CGMacros must have CC BY-NC-SA 4.0
    assert "CC BY-NC-SA 4.0" in notice_content
    assert "CGMacros" in notice_content
    assert "CC BY-NC-SA 4.0" in data_licenses_content
    assert "CGMacros" in data_licenses_content

    # All 5 data sources must be present in DATA_LICENSES.md
    for src in ["CGMacros", "HUPA-UCM", "AI-READI", "Synthea", "IFCT 2017"]:
        assert src in data_licenses_content, f"Missing data source '{src}' in DATA_LICENSES.md"

    # Raw data redistribution must be explicitly No
    assert "No" in data_licenses_content


def test_no_unverified_license_claimed_in_docs():
    notice_text = Path("models/checkpoints/NOTICE.md").read_text(encoding="utf-8")
    data_licenses_text = Path("DATA_LICENSES.md").read_text(encoding="utf-8")

    check_no_unverified_license_claimed(notice_text, "models/checkpoints/NOTICE.md")
    check_no_unverified_license_claimed(data_licenses_text, "DATA_LICENSES.md")

    # Specific rows checks
    assert (
        "seen on the institutional repository record, confirmed on the mendeley data page"
        in data_licenses_text.lower()
    )
    assert (
        "terms not published on the pages reviewed, raw files not redistributed"
        in data_licenses_text.lower()
    )
    assert (
        "raw ingredient nutritional values used with acknowledgment, no tables redistributed"
        in data_licenses_text.lower()
    )
    assert "indian_meal_templates.csv" in data_licenses_text
    assert "ifct_foods.csv" not in data_licenses_text

    # The words 'verified' or 'confirmed' next to a license are allowed for CGMacros and HUPA-UCM only
    for line in data_licenses_text.splitlines():
        if "AI-READI" in line or "Synthea" in line or "IFCT" in line:
            for lic in [
                "MIT",
                "Apache",
                "BSD",
                "GPL",
                "Creative Commons",
                "CC BY",
                "CC-BY",
                "ODbL",
            ]:
                pat = re.compile(
                    rf"\b{re.escape(lic)}\b.*?\b(verified|confirmed)\b|\b(verified|confirmed)\b.*?\b{re.escape(lic)}\b",
                    re.IGNORECASE,
                )
                assert not pat.search(line), (
                    f"Found verified or confirmed next to license on line: {line}"
                )


def test_repo_hygiene_ifct_book_and_extract_not_tracked_by_git():
    import subprocess

    cmd = ["git", "ls-files", "data/external/ifct_book", "data/interim/ifct_extract.csv"]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    tracked_files = [line.strip() for line in res.stdout.splitlines() if line.strip()]
    assert len(tracked_files) == 0, f"Sensitive or raw data tracked in git: {tracked_files}"


def test_license_validator_fails_on_unverified_claim():
    # Negative test ensuring the validator catches and fails on invalid claims
    malformed_sample_1 = "Synthetic AI-READI | MIT license (verified on Synthetix) | No raw data"
    with pytest.raises(ValueError, match="found 'verified' or 'confirmed' next to license"):
        check_no_unverified_license_claimed(malformed_sample_1, "mock_DATA_LICENSES.md")

    malformed_sample_2 = "Synthea | Apache 2.0 (confirmed before use) | No raw data"
    with pytest.raises(ValueError, match="found 'verified' or 'confirmed' next to license"):
        check_no_unverified_license_claimed(malformed_sample_2, "mock_DATA_LICENSES.md")

    malformed_sample_3 = "IFCT 2017 food values | CC BY 4.0 confirmed | No raw data"
    with pytest.raises(ValueError, match="found 'verified' or 'confirmed' next to license"):
        check_no_unverified_license_claimed(malformed_sample_3, "mock_DATA_LICENSES.md")
