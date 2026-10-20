import sys
import os

sys.path.insert(0, os.path.abspath("."))

import json
from src.glucotwin.serving.app import v1_app

v1_schema = v1_app.openapi()

paths = {}
for k, v in v1_schema.get("paths", {}).items():
    if k == "/health":
        paths["/health"] = v
    else:
        paths[f"/api/v1{k}"] = v

full_schema = {
    "openapi": v1_schema.get("openapi", "3.1.0"),
    "info": v1_schema.get("info", {}),
    "paths": paths,
    "components": v1_schema.get("components", {}),
}

with open("frontend/lib/openapi.json", "w", encoding="utf-8") as f:
    json.dump(full_schema, f, indent=2)

print(
    "Exported OpenAPI with", len(full_schema.get("components", {}).get("schemas", {})), "schemas."
)
