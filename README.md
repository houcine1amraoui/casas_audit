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
Reads each selected home's raw file (tiers in `audit.tiers`, default `core_single`) inside its stable window.
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
