from typing import Any


def import_synthea_bundle(bundle_json: dict[str, Any]) -> list[dict[str, Any]]:
    # Synthea FHIR bundle extractor for patient demographics and observations
    extracted_records: list[dict[str, Any]] = []
    entries = bundle_json.get("entry", [])
    for entry in entries:
        resource = entry.get("resource", {})
        if resource.get("resourceType") == "Patient":
            extracted_records.append(
                {
                    "id": resource.get("id", "UNKNOWN"),
                    "gender": resource.get("gender", "male"),
                    "birthDate": resource.get("birthDate", "1980-01-01"),
                }
            )
    return extracted_records
