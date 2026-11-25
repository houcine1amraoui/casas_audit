# CASAS free-living audit

Dataset: CASAS Smart Home dataset, free living (Zenodo record 15708568, mapped sensor names, CC BY 4.0).
Cite: Cook, Crandall, Thomas & Krishnan (2013), CASAS: A smart home in a box, IEEE Computer 46(7):62-69.

## Step 1: inventory (`python -m run_inventory`)
Source (`source` in `configs/defaults.yaml`): `auto` uses already unzipped `data/`, `labeled_data/`, `floorplans/` folders under
the dataset folder (searched up to 3 levels down, no download); otherwise the zips. In zips mode it downloads `data.zip`, `floorplans.zip`, `labeled_data.zip` into the dataset folder (md5-checked, skipped when already
present) and the record metadata. Then, per home file, streamed from the zip (inner .zip extracted to local disk,
.gz/.bz2 read compressed): line layout, unparsable lines with examples, time span, timestamps going backwards, exact
duplicate events, event-free gaps, active days (>= 50 events), longest active run, sensors, message vocabulary,
sensor kind guessed from messages, extra fields (labels). Nothing is filtered; `homes.csv` marks *candidate* homes
by a proposed rule (`selection` in `configs/defaults.yaml`). Residents per home come from the record's description
table (`zenodo_description.txt`, `zenodo_table_*.csv`), not from the files.

Outputs (`outputs/casas_inventory/`): `inventory_report.md`, `homes.csv`, `sensors.csv`, `days.csv.gz`, `gaps.csv`,
`labeled_homes.csv`, `labeled_sensors.csv`, `floorplans.csv`, record description files, `run.log`.

Test without network: `python tests/make_synthetic.py /tmp/syn` then
`python -m run_inventory --project_root_dir /tmp/root --dataset_folder /tmp/syn --set zenodo.download=false --set zenodo.verify_md5=false --set paths.local_dir=/tmp/local`.

Partial rerun: `--set "inventory.parts=[labeled,floorplans]"` adds labels and floorplans to an existing data inventory.
If `--dataset_folder` points at `data/` itself, `labeled_data/` and `floorplans/` are looked for beside it.

## Step 2: structure, duplicates, cohort (`python -m run_select --project_root_dir ROOT`)
Reads the step-1 outputs only. Parses sensor names into a hierarchy (`<room type><letter><spot>`: spot sensor -> room ->
home; doors/entryways/windows -> perimeter), detects the same physical home under two ids (identical daily event counts),
takes residents per home from the record table, finds each home's stable window (span where every core in-room motion
sensor reports) and proposes tiers: `core_single` (one resident: the cross-user cohort), `multi_resident`,
`unknown_residents`, `excluded` (with the reason). Outputs in `outputs/casas_selection/`.

## Step 3: audit of the selected homes (`python -m run_audit --project_root_dir ROOT --dataset_folder DATA`)
Reads each selected home's raw file (tiers in `audit.tiers`, default `core_single`) inside its stable window, then finds a
*clean sub-window* (longest run of days on which >= 90% of the home's core motion sensors fire, capped at 180 days).
Integrity and sensor classes are reported for both; features and tests use only the clean days. Doors are not judged
`stuck` (they can stay open); numeric sensors (temperatures) are not streams.
1. Integrity: daily activations, event-free stretches >= 12 h and whether a door/entry event preceded them (absence)
   or not (likely outage), backwards timestamps, ignored message types.
2. Sensors: ON/OFF pairing, ON durations, stuck sensors, dropouts on active days, rare/silent/chattering streams.
3. Within-home consistency: first vs last 28 active days (permutation test, drift, weekday routine).
4. Cross-home distinguishability: profile from the first 40% of a home's days, test on the last 40%; nearest-profile
   identification with `time_only` (layout-free), `room_groups` and both; also within each deployment (e.g. all `tm`).
5. Labels: activity segments from both label styles; share of activations by room group per activity (sanity check of
   the room parsing). Daily features per home are cached in `cache/casas/features/<home>.npz` for later modelling.
Synthetic check: `python tests/make_synthetic_homes.py /tmp/s` creates homes with a stuck sensor, a dead sensor, an
absence, an outage, a routine shift, a duplicate home and both label styles.

## Step 4: Stage E, per-home models with cross-user deviation (`python -m run_stage_e --project_root_dir ROOT`)
Needs the step cache written by run_audit (`cache/casas/steps/<home>.npz`). For each home with enough clean days:
10-min activation counts per usable sensor, chronological split of clean days (train 60% / val 15% / test 25%),
models persistence, seasonal naive, ridge VAR, LSTM, LSTM autoencoder, hierarchical autoencoder (HAE: sensor -> room ->
home codes; controls `hae_random` with sensors assigned to random rooms of the same sizes and `hae_flat` with a single
room of the same total code size), GDN without graph, GDN, GDN + room hierarchy, GDN + randomly assigned rooms. Score
settings (channel aggregation, smoothing, and calibration `none` or `hour` = scores standardised per 2-hour bin of the
day using clean val windows) are chosen on anomalies injected into val. Evaluation on held-out data: forecast error, injected anomalies in test, and the cross-user
test: the home's model scores its own test days and other homes' test days mapped onto its sensors by name; AUROC per
home. Flat vs hierarchical is compared per home (seeds averaged) with bootstrap CIs and Wilcoxon tests over homes.
Results are saved per (home, model, seed) under `outputs/<run_name>/runs/` (v2 run: `outputs/casas_stage_e_v2/`; v1 results stay in `outputs/casas_stage_e/`), so a run can be stopped and resumed, or
split with `--set "stage_e.homes=[...]"`; `--report_only` rebuilds the report. `--quick` = smoke test.
