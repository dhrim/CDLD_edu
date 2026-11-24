# Source data and locally reconstructed evaluation inputs

## EdNet

Provider: Riiid. Source: https://github.com/riiid/ednet .
Reference: Choi et al., *EdNet: A Large-Scale Hierarchical Dataset in Education*.
License: Creative Commons Attribution-NonCommercial 4.0 International:
https://creativecommons.org/licenses/by-nc/4.0/ .

The EdNet arrays here are processed extracts: evaluation response labels, item
categories, pairing indices, response counts and training item proportions were
selected, indexed or calculated for the study and stored as NPZ files. Fitted
model predictions were added by the study authors. The source-derived material
remains subject to the provider's attribution and noncommercial conditions.
No provider endorsement is implied.

## Eedi

Provider: Eedi, NeurIPS 2020 Education Challenge.
Source and download: https://www.eedischool.com/projects/neurips-education-challenge .
Reference: Wang et al. (2020), *Diagnostic Questions: The NeurIPS 2020 Education Challenge*.
License: Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International:
https://creativecommons.org/licenses/by-nc-nd/4.0/ .

Obtain the official data.zip from the provider under its terms. The current
package contains fitted model predictions, but does not bundle Eedi response or
category labels, source student/item indices, per-student response counts or
training item proportions. `prepare_eedi_labels.py` reconstructs these inputs
locally from the original Tasks 1 and 2 training data and question/subject
metadata. No question images or student demographics are used. The script does
not fit models. Generated `local_data/` is excluded from Git; do not redistribute
it as part of this package. Per-array checksums verify reconstruction without
publishing the corresponding records.

The provider limits redistribution of modified source material. This separation
does not relicense Eedi data or confer additional permissions. Previous Git
commits may contain the earlier combined archives; use the current package.

## Code and analysis outputs

Source-dataset licenses are not blanket licenses for the authors' code, and code
licensing does not override third-party data conditions. Aggregate published
statistics and the executed notebook remain available. Locally generated
training/export caches may contain source-derived data and are not release
artifacts. Do not apply a single permissive license to the entire directory.
