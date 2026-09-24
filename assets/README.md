# Large artifacts

`manifest.json` lists required model, data, calibration, and reset assets. Every published artifact must have a measured byte length and SHA-256 digest. Source paths are resolved under `TCR_WORKSPACE_ROOT`; they do not make this public Git checkout self-contained.

Weights above GitHub's file-size limit will be split into release assets below 2 GB with a chunk manifest and reassembled with SHA-256 verification. Until the release is actually uploaded and the URL is recorded here, the manifest reports the asset as **not published**. Do not rename, resize, or quantize a checkpoint to make it fit; that would change the evaluated model.
