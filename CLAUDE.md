# Project Guidance

## Profiling / Neuron Explorer

When profiling NKI kernels, ensure the Neuron profile server is running BEFORE ingesting; profiles register under the 'Search Profiles' tab (not the main page) once status is PROCESSED. Avoid `--ingest-only` without a running server.

## Style / Communication

Keep code comments and explanations concise; do not narrate irrelevant bug-fixing steps. Prefer minimal, contract-preserving fixes over hacky monkey-patches.

## Neuron SDK

For Neuron SDK version issues, distinguish the SDK umbrella version from individual package versions (neuronx-cc, neuronx-distributed), and verify a target version actually exists in the repo before suggesting an upgrade.
