# Training and prediction-generation source

This directory supplies the study's preprocessing, split construction, CDLD Finder,
SVD, nested item classification, MLP, and bilinear prediction code. The original
scripts are under `legacy/`; `source_manifest.json` identifies their sources and
SHA-256 checksums. `run.py` adds portable path setup and connects these scripts.
It is a newly assembled runner, not a claim that a complete retraining run was
performed again when preparing this release.

Use the repository-root `reproduce.py` for the paper's exact archived numerical
results. This directory regenerates fitted predictions from provider data.
Retraining on a different runtime/GPU can change fitted values; new results are
exported separately and are not required to equal the archived results to 1e-10.

## Environment

The original October training environment used Linux, Python 3.12.15,
TensorFlow 2.21.0, Keras 3.15.1, NumPy 2.4.6, SciPy 1.17.1,
scikit-learn 1.9.1, and RTX A5000 GPUs. See `environment-original.txt`.
`requirements.txt` pins the principal packages from that environment.
Use a separate environment from the repository-root numerical-reproduction
requirements. The historical Finder/MLP scripts require an available GPU.
The runner executes jobs sequentially on one selected GPU; it does not provision
servers. Full provider datasets require substantial RAM and disk space.

```sh
python3.12 -m venv .venv-training
. .venv-training/bin/activate
pip install -r training/requirements.txt
python training/run.py --help
```

## Inputs

Obtain datasets directly from their providers, under their terms:

- EdNet: https://github.com/riiid/ednet — `EdNet-KT1.zip` and Contents `questions.csv`.
- Eedi: https://www.eedischool.com/projects/neurips-education-challenge — official
  `data.zip`, containing `data/train_data/train_task_1_2.csv`,
  `data/metadata/question_metadata_task_1_2.csv`, and
  `data/metadata/subject_metadata.csv`.

No participant data are bundled with the training code. Use a fresh work directory
outside this repository, on a filesystem supporting symbolic links. Keep its parent
folder dedicated to this run because the original Eedi script uses a sibling
`자료/Eedi/metadata` link; the runner creates this link. Existing conflicting links
or modified source copies cause an error instead of being overwritten.

## Execution order

From the repository root, replace the example absolute paths below. The same work
directory is shared by both datasets. Default seeds are 42–46.

```sh
python training/run.py prepare --dataset EdNet --work /data/cdld-run/work --raw /data/EdNet-KT1.zip --questions /data/questions.csv
python training/run.py prepare --dataset Eedi --work /data/cdld-run/work --raw /data/eedi/data.zip

for dataset in EdNet Eedi; do
  python training/run.py split --dataset "$dataset" --work /data/cdld-run/work
  python training/run.py finders --dataset "$dataset" --work /data/cdld-run/work --gpu 0
  python training/run.py features --dataset "$dataset" --work /data/cdld-run/work
  python training/run.py classify --dataset "$dataset" --work /data/cdld-run/work
  python training/run.py predictors --dataset "$dataset" --work /data/cdld-run/work --variant raw-rate --gpu 0
  python training/run.py predictors --dataset "$dataset" --work /data/cdld-run/work --variant no-rate --gpu 0
  python training/run.py predictors --dataset "$dataset" --work /data/cdld-run/work --variant logit-rate --gpu 0
done
python training/run.py classify --dataset EdNet --mode bundle --work /data/cdld-run/work
python training/export.py --work /data/cdld-run/work --output /data/cdld-run/new-results --analyze
```

`features` fits IRT for the EdNet classification baseline, builds the response and
exposure SVD coordinates and applies the original scaling. Eedi IRT is fitted by
the original classification script. EdNet common folds require all five seeds;
Eedi's original split script also creates all five splits. `classify` uses the
20,000-iteration continuation for EdNet item seeds 42/46, matching the recorded
convergence reruns; it retains the same tolerance, folds and regularization grid.

`raw-rate` is the main shared-item-proportion comparison. `no-rate` produces the
secondary no-rate fits; EdNet exposure MLP includes its original extended run.
`logit-rate` fits bilinear predictors only. MLP selection uses validation AUC;
bilinear selection uses validation loss and extends the regularization grid when
the selected value reaches its upper boundary. Export checks row/label alignment
before joining predictions. The exported files can be analyzed with the original
metric/bootstrap functions without replacing the published predictions.

Do not use the historical launch/status scripts from unrelated experiments.
They are not needed here. Several archived core definitions contain unused cold-start
helpers inherited from the original EdNet implementation; only the warm split is
selected by this runner. No KMLE data or experiments are included.

## Checks and provenance

```sh
python training/check.py
python training/legacy/20261028_revision1/EdNet/bilinear_fast.py --selftest
python training/legacy/20261028_revision1/Eedi/bilinear_fast.py --selftest
python training/legacy/20261029_bilinear_logit_rate/EdNet/bilinear_fast.py --selftest
python training/legacy/20261029_bilinear_logit_rate/Eedi/bilinear_fast.py --selftest
```

The CPU check validates original file hashes and Python syntax, the original EdNet
split on synthetic responses, and Eedi preprocessing, query exclusion, SVD,
scaling, category folds and filesystem connections on a synthetic official-layout
archive. Its Finder vectors are explicitly random fixtures, not fitted results.
The bilinear commands require TensorFlow and independently compare the implemented
loss/gradients with NumPy formulas. They are provided for the training runtime.

Release validation: the CPU check and archived numerical reproduction were run;
full GPU retraining was not run. The mathematical/model definitions were compared
with the archived EdNet notebook. The numerical-reproduction package retains the
original saved predictions and reference values unchanged.
