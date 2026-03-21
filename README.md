# Data Processing

Tools for converting raw recorded HDF5 episodes into training-ready datasets.
Compatible with openpi and other robot-learning frameworks (LeRobot, ACT, Diffusion Policy, etc.).

## Quick Start

```bash
# 1. Check environment (Python 3.10+, pip packages)
bash setup_env.sh

# 2. Run the full pipeline
bash pipeline/pipeline.sh path/to/raw_dataset_dir

# 3. Custom target size or resize method
bash pipeline/pipeline.sh path/to/raw_dataset_dir --size 224 --method center_crop
```

## Directory Structure

```
data_processing/
├── setup_env.sh            # Check and install dependencies
├── requirements.txt        # Python package list
│
├── pipeline/               # Sequential processing steps
│   ├── 01_check_dataset.py
│   ├── 02_drop_front_camera.py
│   ├── 03_smooth_episodes.py
│   ├── 04_trim_episodes.py
│   ├── 05_resize_images.py
│   └── pipeline.sh         # Orchestrates all 5 steps end-to-end
│
└── viz/                    # Standalone visualisation tools (no data modified)
    ├── viz_episode.py
    └── viz_trajectory.py
```

## Typical Workflow

```
raw episodes (480×480)
    │
    ├── [optional] viz/viz_episode.py      ← review & delete bad episodes
    │
    ▼  pipeline/pipeline.sh
    1. check_dataset        quality gate
    2. drop_front_camera    keep exterior + wrist only  [optional]
    3. smooth_episodes      Savitzky-Golay on qpos / action
    4. trim_episodes        cut idle frames at start/end
    5. resize_images        480×480 → 224×224 (or any target size)
    │
    ├── [optional] viz/viz_trajectory.py   ← verify smoothing / trimming
    │
    ▼
training_dataset/ (224×224, training-ready)
```

---

## Pipeline Scripts

### `pipeline/01_check_dataset.py` — quality report

Scans all episodes and prints a report of data issues. No files are written.

```bash
python3 pipeline/01_check_dataset.py path/to/dataset_dir
python3 pipeline/01_check_dataset.py path/to/dataset_dir --spike-thresh 0.10
```

| Flag | Meaning |
|------|---------|
| `cameras` | Number of camera streams detected (informational) |
| `static_action` | `action` never changes — likely a recording bug |
| `frozen_gripper` | Gripper dimension is constant for the whole episode |
| `qpos_eq_action` | `qpos ≈ action` — recorder may be duplicating state |
| `short` | Fewer than `--min-steps` timesteps |
| `spikes` | Joint step exceeds `--spike-thresh` rad (warning, not failure) |

Exit code is `1` if any structural issues are found, `0` otherwise.

---

### `pipeline/02_drop_front_camera.py` — remove front camera stream

Copies all episodes to a new directory, dropping `front_image_1` so downstream
tools only see the exterior and wrist cameras. Skip this step with `--keep-front`
in `pipeline.sh` when three cameras are needed.

```bash
python3 pipeline/02_drop_front_camera.py path/to/raw_dir           path/to/no_front_dir
python3 pipeline/02_drop_front_camera.py path/to/episode_0.hdf5    path/to/episode_0_nf.hdf5
```

---

### `pipeline/03_smooth_episodes.py` — trajectory smoothing

Applies Savitzky-Golay smoothing to `qpos` and `action` trajectories.
The gripper dimension is left unsmoothed to preserve open/close transitions.
Run **before** trimming so the smoother doesn't see pad frames at the edges.

```bash
python3 pipeline/03_smooth_episodes.py path/to/dataset_dir        smoothed/
python3 pipeline/03_smooth_episodes.py path/to/episode_0.hdf5     smoothed/episode_0.hdf5
python3 pipeline/03_smooth_episodes.py path/to/dataset_dir        smoothed/ --window 9 --poly 2
```

---

### `pipeline/04_trim_episodes.py` — cut start/end frames

**Interactive mode** (default) — loops through each episode, shows the frame
count, and asks you to type the keep range:

```bash
python3 pipeline/04_trim_episodes.py path/to/dataset_dir  trimmed/
```

```
  episode_0.hdf5  [250 frames]
  Keep range (start end, negative ok, Enter = keep all): 10 -8
  → keeps frames 10-242  (233 frames)
```

- `10 230`  — keep frames 10 to 230 (inclusive)
- `10 -5`   — keep frames 10 to T−5 (negative counts from end)
- Enter      — keep all frames unchanged

**Batch mode** — apply the same range to every episode:

```bash
python3 pipeline/04_trim_episodes.py path/to/dataset_dir  trimmed/ --start 10 --end -8
```

**Dry-run** — preview the plan without writing anything (omit dst):

```bash
python3 pipeline/04_trim_episodes.py path/to/dataset_dir
```

---

### `pipeline/05_resize_images.py` — resize images to training resolution

Converts all camera streams in HDF5 episodes from the recorded resolution
(e.g. 480×480) to the target training resolution (default 224×224).
Three preprocessing strategies are available:

| `--method` | Behaviour | Use when |
|---|---|---|
| `center_crop` (default) | Square-crop centre → resize | Task object always centred |
| `resize` | Squish full image to target | Global context matters |
| `pad_resize` | Aspect-ratio resize + black padding | Source is not square |

```bash
# Default: 224×224 center crop  (pi0.5 / ACT)
python3 pipeline/05_resize_images.py trimmed/              training_dataset/

# Single file
python3 pipeline/05_resize_images.py episode_0.hdf5        episode_0_224.hdf5

# Custom size and method
python3 pipeline/05_resize_images.py trimmed/              training_dataset_256/ --size 256 --method resize

# Dry-run (omit dst)
python3 pipeline/05_resize_images.py trimmed/ --dry-run
```

Trajectory data (`qpos`, `action`) is copied unchanged. Parallelised over
episodes via `--workers` (default 4).

---

### `pipeline/pipeline.sh` — end-to-end pipeline

Runs all 5 steps sequentially with confirmation prompts between stages.

```bash
./pipeline/pipeline.sh path/to/raw_dataset_dir
./pipeline/pipeline.sh path/to/raw_dataset_dir --size 224 --method center_crop
./pipeline/pipeline.sh path/to/raw_dataset_dir --keep-front --size 256 --method resize
```

| Option | Default | Description |
|--------|---------|-------------|
| `--out DIR` | `training_dataset` | Final output directory |
| `--size N` | `224` | Target image size in pixels |
| `--method STR` | `center_crop` | Resize strategy |
| `--window N` | `15` | SG filter window (must be odd) |
| `--poly N` | `3` | SG polynomial order |
| `--cuts FILE` | `cuts.json` | Per-episode cut ranges from `viz_trajectory.py` |
| `--trim N` | `0` | Global fallback frames to cut each end |
| `--keep-front` | — | Skip the drop_front_camera step |
| `--workers N` | `4` | Parallel workers for resize step |

Intermediate directories created:
```
raw_dataset_no_front/   after step 2
raw_dataset_smoothed/   after step 3
raw_dataset_trimmed/    after step 4
training_dataset/       final output (step 5)
```
Intermediates are safe to delete once the pipeline completes successfully.

---

## Visualisation Tools

These tools never modify data and can be run at any point independently.

### `viz/viz_episode.py` — video playback viewer

Plays back all camera streams side-by-side (2 or 3 cameras, auto-detected) with
per-joint trajectory strips and a scrubbing progress bar.

```bash
python3 viz/viz_episode.py path/to/dataset_dir
python3 viz/viz_episode.py path/to/episode_0.hdf5 --fps 30 --scale 2.0
```

| Key | Action |
|-----|--------|
| `SPACE` | Pause / resume |
| `←` / `→` | Step ±5 frames |
| `↑` / `↓` | Previous / next episode |
| `F` | Toggle 2× speed |
| `R` | Restart |
| `D` | Arm delete (shows red confirmation banner) |
| `Y` | Confirm delete — removes file from disk |
| `Q` | Quit |

Mouse drag on the progress bar or trajectory strips to scrub.

---

### `viz/viz_trajectory.py` — original vs processed trajectory comparison

Overlays original and processed joint trajectories in 7 vertical subplots —
one per joint. Useful for verifying smoothing and trim decisions.

```bash
python3 viz/viz_trajectory.py original_dir/ training_dataset/
python3 viz/viz_trajectory.py original_dir/ training_dataset/ --no-norm
```

`--no-norm` plots absolute frame numbers instead of a normalised 0–1 x-axis.

| Key | Action |
|-----|--------|
| `↑` / `←` / `P` | Previous episode |
| `↓` / `→` / `N` | Next episode |
| `S` | Save figure as PNG |
| `Q` / `Escape` | Quit |
