# PINN datasets — provenance & download

Offline preprocessing inputs for the `applications/pinn` experiments. Raw data is
**not** committed (bulky, redistributable-but-large); only small **derived**
artifacts are committed via Git LFS. This file documents exactly how to obtain the
raw inputs so the derived artifacts and paper results are reproducible.

## NGSIM I-80 (real-data validation, Paper 1 §6.5)

Source: **Next Generation Simulation (NGSIM) Vehicle Trajectories and Supporting
Data** — U.S. DOT, public domain. Vehicle trajectories on I-80 (Emeryville, CA),
sampled at 10 Hz.

### Automated download (recommended)

A helper fetches the I-80 zip, extracts the chosen period's trajectory **CSV**
(the header-less space-delimited `.txt` sibling is skipped), and stages it:

```bash
uv run python -m applications.pinn.data.download_ngsim   # -> raw/i80.csv (0500-0515)
```

Stdlib-only; downloads to `raw/.ngsim-cache/` (zip reused on re-run). Needs
internet on the host running it. The download URL is deterministic (dataset
`8ect-6jqj`, I-80 asset `ea269540-b86c-4b2d-a9c2-c8f4c0a3d0a0`); override with
`--url` if the Socrata blobstore path changes, or `--period {0400-0415,0500-0515,
0515-0530}`.

### Manual download

Download on a networked machine (the dev container's egress may be firewalled),
then copy the file into the container workspace.

1. **Official — ITS DataHub (`data.transportation.gov`).**
   - Web UI: open the *"Next Generation Simulation (NGSIM) Vehicle Trajectories
     and Supporting Data"* dataset, **Filter** on `Location = i-80`, then
     **Export -> CSV**. (The full multi-site file is ~1.5 GB; filter to I-80
     first.)
   - API (Socrata/SoQL) — you must pass `$limit` (default is 1000 rows) and
     ideally an app token for a pull this size:
     ```
     https://data.transportation.gov/resource/<RESOURCE_ID>.csv?location=i-80&$limit=50000000
     ```
     Confirm `<RESOURCE_ID>` on the dataset page (the *trajectories* resource).
2. **Fallback:** public Kaggle / GitHub mirrors of the I-80 trajectories exist;
   verify provenance and note it if used.

I-80 has three 15-minute windows (4:00-4:15, 5:00-5:15, 5:15-5:30 pm). Any one
works; the **5:00-5:15 pm** congested period has the richest stop-and-go waves,
best for the monotone-window scan.

### Where to put it

```
applications/pinn/data/raw/i80.csv      # gitignored; keep local
```

### Expected schema & units

`applications/pinn/data/ngsim.py::_load_raw` reads a CSV **header row** with these
exact column names:

```
Vehicle_ID, Global_Time, Local_Y, v_Vel, Lane_ID
```

- `Global_Time` — epoch milliseconds (the loader converts ms -> s).
- `Local_Y` — longitudinal position along the road.
- **Units caveat:** original NGSIM `Local_Y` / `v_Vel` are in **feet / ft·s⁻¹**.
  The loader treats `Local_Y` as **metres**. Either convert to SI before running
  (`Local_Y *= 0.3048`, `v_Vel *= 0.3048`) for a clean metric FD, or run as-is and
  know the derived FD, `--dx`/`--dt`, and ASM parameters are then in feet-based
  units (physics ratios still hold; only absolute scales change).
- If your export uses different column names/capitalization, rename them or adjust
  `_load_raw` (a two-line change).

Verify before running:

```bash
head -1 applications/pinn/data/raw/i80.csv
```

### Build the derived artifact

See [`../RUNBOOK.md`](../RUNBOOK.md) ("Real-data validation") for the full
sequence. In short:

```bash
uv run python -m applications.pinn.data.ngsim \
    --raw applications/pinn/data/raw/i80.csv \
    --out applications/pinn/data/ngsim-i80-wave.npz
```

This writes `ngsim-i80-wave.npz` (committed via Git LFS): the Edie density field on
the chosen monotone window, the calibrated Greenshields FD, `sign_x`, and the
reported `monotonicity_defect`. **Gate:** inspect `monotonicity_defect` and the
window heatmap before proceeding (see RUNBOOK).
