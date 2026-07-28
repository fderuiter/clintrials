# Trial Simulation Hub

This directory contains the dynamic frontend assets for the Trial Simulation Hub.

## Local Development

Before serving `index.html` locally, make sure to generate the `schema.json` artifact by running the serialization script from the repository root:

```bash
poetry run python scripts/serialize_schemas.py
```

This ensures that the latest Python schemas are serialized to `hub/schema.json` and are available to the frontend.
