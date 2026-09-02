# Artifacts

Public checkpoints and demonstration files are indexed by logical name in
`midas/artifacts_index.json`. `midas.utils.artifacts.download_artifact` downloads
to a temporary `.part` file, verifies SHA-256, and only then publishes it.

The v0.1.0 implementation does not bundle or advertise trained checkpoints.
Future records must include a stable URL, SHA-256 digest, license/provenance,
training configuration, and expected evaluation metric before release.
