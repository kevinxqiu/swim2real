# Paper-number reference

- `paper_numbers.json` maps each stable scalar key to its recomputed value,
  definition, paper location and independently transcribed printed values.
- `paper_numbers.md` renders the same information and lists printed values
  that require separate recording or hardware measurements.
- `source_checks.json` contains 571 scalar expectations read from the original
  analysis summaries, with each source summary's SHA-256.
- `archive_manifest.json` identifies the compact record archive by size and
  SHA-256. The [data card](../dataset/README.md) documents its contents.

Run `swim2real reproduce --records DIR --out-json numbers.json --out-md numbers.md`,
then `swim2real check-numbers numbers.json`. From an uninstalled checkout,
`PYTHONPATH=src python -m swim2real.analysis.cli` provides the same commands.
The only analysis runtime dependency is NumPy. SciPy is also supported in
the analysis environment; the exact signed-rank implementation does not
depend on a library's asymptotic-test defaults.

The original fifteen-seed summary contains fourteen completed direct-update
runs. The paper's later fifteen-run descriptive set is checked against its
own result manifest and descriptives. Interpolation uses the 32 terminal
assessment records; its training histories are not included.

To run the full archive check, set `SWIM2REAL_RECORDS` to the extracted
directory or ZIP, or place `runs.zip` here, and run `pytest -q tests/analysis`.
Without the archive, the full test skips explicitly and the fixture tests
still run. Set `PYTHONPATH=src` if the package has not been installed.
