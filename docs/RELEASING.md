# Releasing

Before a public tag: obtain MIT and dataset/artifact rights-holder sign-off;
run the CPU test/build suite in a clean clone; run the LIBERO and RoboCasa GPU
runbooks; scan the repository and submodules for secrets/private paths; verify
artifact licenses and hashes; and test an anonymous recursive clone. Build with
`python -m build`, inspect wheel/sdist contents, then tag only the validated
commit. Research checkpoints from before the MIDAS rename are unsupported.
