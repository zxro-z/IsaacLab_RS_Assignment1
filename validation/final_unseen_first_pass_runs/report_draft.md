# Frozen Final: first-pass / terrain-section analysis

## A. Analysis definition

Post-hoc measurement only. Existing Final artifacts contained final metrics but no position trajectories; four frozen 100-env first-episode runs were instrumented. Seed 24, terrain seed 2404, same checkpoints and policy observation adapters. All 400 episodes matched original return, length, legacy final displacement and fall/timeout within rtol=1e-6, atol=1e-6 (steps/flags exact). Original summaries remain unchanged.

Boundaries were read from the runtime COURSE_DESIGN and frozen source: Flat 0–4; ridge 4–10; terraces 10–18 (subranges 10–12.2, 12.2–15.6, 15.6–18); basin 18–25; landing 25–40 m.

The terrain generator centers the full 400×80 m layout at world (0,0); tile (row,col) starts at (40*row-200, 8*col-40). Its env origin is tile-start + (2,4,0). Root reset adds robot default (0,0,0.5) to that origin. Runtime initial XY confirms local (2,4) for every episode. We use x_local = root_world_x - env_origin_x + 2, always relative to the INITIAL tile and never modulo 40. Consequently local boundaries 4/10/18/25/40 correspond to initial forward displacement 2/8/16/23/38 m. Flat traversal starts at local x=2, so it covers the remaining spawn flat, not the unvisited x=0–2.

Physics dt=0.00833333333333333 s, decimation=2, runtime control dt=0.0166666666666667 s. First-pass is the first sampled x_local >= boundary, inclusive of pre-reset terminal pose. Time=step*control_dt; no interpolation. Sample timing resolution is one control step. Backtracking does not undo reach.

The legacy evaluator records displacement from step N-1 on a terminal transition. This was preserved and checked separately. First-pass uses true step-N terminal position, captured through the existing pre-reset callback without extra observation/sensor computations or recorder terms.

One Base episode (env 94, step 960) has both terminated and truncated true. The frozen TerminationManager.compute clears the per-term table when each later true term fires, so get_term(time_out) is false after torso_height fires. The existing evaluator therefore labels it fall only. Original flags are preserved in the raw timeout/fall columns; actual terminal flags are separate terminated/truncated columns. Timeout localization uses actual truncated, yielding Base 82 timeouts (one overlapping fall), versus the original exclusive-label count 81. This is documented classification behavior, not a rerun inconsistency.

Wilson 95% intervals are shown for rates. Time std uses ddof=0; percentile bootstrap 95% mean and median intervals use 10,000 resamples, RNG seed 240403. Times are conditional on reaching/completing, so survivor selection affects comparisons. These intervals describe episode variation within one fixed scene, not training-seed/terrain-seed generalization.

## B. Boundary reach table

| Policy | ≥4 m | ≥10 m | ≥18 m | ≥25 m | ≥40 m |
|---|---:|---:|---:|---:|---:|
| Base | 97/100 (97%; 91.5–99.0) | 96/100 (96%; 90.2–98.4) | 93/100 (93%; 86.3–96.6) | 93/100 (93%; 86.3–96.6) | 92/100 (92%; 85.0–95.9) |
| Original HeightScan | 98/100 (98%; 93.0–99.4) | 97/100 (97%; 91.5–99.0) | 97/100 (97%; 91.5–99.0) | 96/100 (96%; 90.2–98.4) | 94/100 (94%; 87.5–97.2) |
| HeightScan+Contact | 96/100 (96%; 90.2–98.4) | 96/100 (96%; 90.2–98.4) | 95/100 (95%; 88.8–97.8) | 94/100 (94%; 87.5–97.2) | 94/100 (94%; 87.5–97.2) |
| Diverse HeightScan+Contact | 97/100 (97%; 91.5–99.0) | 95/100 (95%; 88.8–97.8) | 93/100 (93%; 86.3–96.6) | 91/100 (91%; 83.8–95.2) | 85/100 (85%; 76.7–90.7) |

Cells: n/100 (percentage; Wilson 95% interval). These are local-X milestone reach rates; lateral exits are reported below.

## C. Conditional section completion

| Policy | Ridge given Flat | Terraces given Ridge | Basin given Terraces | Landing given Basin |
|---|---:|---:|---:|---:|
| Base | 96/97 (99%; 94.4–99.8) | 93/96 (97%; 91.2–98.9) | 93/93 (100%; 96.0–100.0) | 92/93 (99%; 94.2–99.8) |
| Original HeightScan | 97/98 (99%; 94.4–99.8) | 97/97 (100%; 96.2–100.0) | 96/97 (99%; 94.4–99.8) | 94/96 (98%; 92.7–99.4) |
| HeightScan+Contact | 96/96 (100%; 96.2–100.0) | 95/96 (99%; 94.3–99.8) | 94/95 (99%; 94.3–99.8) | 94/94 (100%; 96.1–100.0) |
| Diverse HeightScan+Contact | 95/97 (98%; 92.8–99.4) | 93/95 (98%; 92.6–99.4) | 91/93 (98%; 92.5–99.4) | 85/91 (93%; 86.4–96.9) |

## D. First-reach time

Seconds; mean ± population std; median; bootstrap 95% CI for mean. Full median bootstrap CIs are in CSV/JSON.

| Policy | Boundary local x | Time |
|---|---:|---|
| Base | 4 m | 1.052 ± 0.290; median 0.967; mean CI [0.999, 1.115]; n=97 |
| Base | 10 m | 2.504 ± 0.364; median 2.383; mean CI [2.436, 2.582]; n=96 |
| Base | 18 m | 4.741 ± 0.502; median 4.667; mean CI [4.643, 4.845]; n=93 |
| Base | 25 m | 6.444 ± 0.575; median 6.383; mean CI [6.332, 6.563]; n=93 |
| Base | 40 m | 10.192 ± 0.675; median 10.108; mean CI [10.058, 10.336]; n=92 |
| Original HeightScan | 4 m | 1.017 ± 0.279; median 0.925; mean CI [0.966, 1.075]; n=98 |
| Original HeightScan | 10 m | 2.362 ± 0.356; median 2.233; mean CI [2.293, 2.434]; n=97 |
| Original HeightScan | 18 m | 4.266 ± 0.525; median 4.217; mean CI [4.164, 4.375]; n=97 |
| Original HeightScan | 25 m | 5.856 ± 0.669; median 5.733; mean CI [5.725, 5.994]; n=96 |
| Original HeightScan | 40 m | 9.210 ± 0.825; median 9.033; mean CI [9.050, 9.380]; n=94 |
| HeightScan+Contact | 4 m | 1.081 ± 0.240; median 1.000; mean CI [1.035, 1.132]; n=96 |
| HeightScan+Contact | 10 m | 2.497 ± 0.281; median 2.433; mean CI [2.442, 2.554]; n=96 |
| HeightScan+Contact | 18 m | 4.361 ± 0.382; median 4.283; mean CI [4.286, 4.439]; n=95 |
| HeightScan+Contact | 25 m | 5.944 ± 0.375; median 5.875; mean CI [5.871, 6.025]; n=94 |
| HeightScan+Contact | 40 m | 9.416 ± 0.483; median 9.317; mean CI [9.323, 9.513]; n=94 |
| Diverse HeightScan+Contact | 4 m | 0.996 ± 0.285; median 0.883; mean CI [0.942, 1.055]; n=97 |
| Diverse HeightScan+Contact | 10 m | 2.345 ± 0.360; median 2.217; mean CI [2.276, 2.420]; n=95 |
| Diverse HeightScan+Contact | 18 m | 4.091 ± 0.396; median 4.017; mean CI [4.012, 4.172]; n=93 |
| Diverse HeightScan+Contact | 25 m | 5.642 ± 0.469; median 5.617; mean CI [5.547, 5.739]; n=91 |
| Diverse HeightScan+Contact | 40 m | 9.011 ± 0.824; median 8.850; mean CI [8.850, 9.195]; n=85 |

## E. Section traversal time

Seconds, conditional on completion. Flat starts at t=0 at local x=2. Other times subtract the two first-pass milestone times. Incomplete traversals are excluded from time statistics and retained in rate denominators.

| Policy | Section | Time |
|---|---|---|
| Base | Flat | 1.052 ± 0.290; median 0.967; mean CI [1.000, 1.114]; n=97 |
| Base | Ridge | 1.457 ± 0.173; median 1.417; mean CI [1.423, 1.494]; n=96 |
| Base | Terraces | 2.232 ± 0.361; median 2.167; mean CI [2.162, 2.310]; n=93 |
| Base | Basin | 1.703 ± 0.252; median 1.617; mean CI [1.653, 1.755]; n=93 |
| Base | Landing | 3.754 ± 0.451; median 3.667; mean CI [3.664, 3.848]; n=92 |
| Original HeightScan | Flat | 1.017 ± 0.279; median 0.925; mean CI [0.965, 1.076]; n=98 |
| Original HeightScan | Ridge | 1.353 ± 0.232; median 1.283; mean CI [1.309, 1.401]; n=97 |
| Original HeightScan | Terraces | 1.904 ± 0.343; median 1.833; mean CI [1.838, 1.977]; n=97 |
| Original HeightScan | Basin | 1.582 ± 0.345; median 1.483; mean CI [1.518, 1.655]; n=96 |
| Original HeightScan | Landing | 3.372 ± 0.447; median 3.300; mean CI [3.284, 3.466]; n=94 |
| HeightScan+Contact | Flat | 1.081 ± 0.240; median 1.000; mean CI [1.035, 1.131]; n=96 |
| HeightScan+Contact | Ridge | 1.416 ± 0.126; median 1.383; mean CI [1.393, 1.443]; n=96 |
| HeightScan+Contact | Terraces | 1.864 ± 0.205; median 1.783; mean CI [1.825, 1.907]; n=95 |
| HeightScan+Contact | Basin | 1.599 ± 0.114; median 1.567; mean CI [1.578, 1.623]; n=94 |
| HeightScan+Contact | Landing | 3.472 ± 0.305; median 3.367; mean CI [3.414, 3.538]; n=94 |
| Diverse HeightScan+Contact | Flat | 0.996 ± 0.285; median 0.883; mean CI [0.941, 1.055]; n=97 |
| Diverse HeightScan+Contact | Ridge | 1.344 ± 0.180; median 1.283; mean CI [1.310, 1.381]; n=95 |
| Diverse HeightScan+Contact | Terraces | 1.751 ± 0.213; median 1.700; mean CI [1.709, 1.795]; n=93 |
| Diverse HeightScan+Contact | Basin | 1.555 ± 0.246; median 1.467; mean CI [1.507, 1.607]; n=91 |
| Diverse HeightScan+Contact | Landing | 3.367 ± 0.650; median 3.233; mean CI [3.246, 3.517]; n=85 |

## F. Failure localization

Cumulative fall-before-boundary: first-episode fall and no prior reach of that boundary. Denominator is all 100 episodes. A fall after completing the first course is separated from a first-pass failure.

| Policy | Before 4 | Before 10 | Before 18 | Before 25 | Before 40 |
|---|---:|---:|---:|---:|---:|
| Base | 3/100 (3%; 1.0–8.5) | 4/100 (4%; 1.6–9.8) | 7/100 (7%; 3.4–13.7) | 7/100 (7%; 3.4–13.7) | 8/100 (8%; 4.1–15.0) |
| Original HeightScan | 2/100 (2%; 0.6–7.0) | 3/100 (3%; 1.0–8.5) | 3/100 (3%; 1.0–8.5) | 4/100 (4%; 1.6–9.8) | 6/100 (6%; 2.8–12.5) |
| HeightScan+Contact | 4/100 (4%; 1.6–9.8) | 4/100 (4%; 1.6–9.8) | 5/100 (5%; 2.2–11.2) | 6/100 (6%; 2.8–12.5) | 6/100 (6%; 2.8–12.5) |
| Diverse HeightScan+Contact | 3/100 (3%; 1.0–8.5) | 5/100 (5%; 2.2–11.2) | 7/100 (7%; 3.4–13.7) | 9/100 (9%; 4.8–16.2) | 15/100 (15%; 9.3–23.3) |

Noncumulative localization by first uncompleted section endpoint (counts):

| Policy | Flat | Ridge | Terraces | Basin | Landing | After course | Timeout before/after 40 | Lateral exits before first pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base | 3 | 1 | 3 | 0 | 1 | 11 | 0/82 | 3 |
| Original HeightScan | 2 | 1 | 0 | 1 | 2 | 17 | 0/77 | 7 |
| HeightScan+Contact | 4 | 0 | 1 | 1 | 0 | 10 | 0/84 | 1 |
| Diverse HeightScan+Contact | 3 | 2 | 2 | 2 | 6 | 18 | 0/67 | 5 |

## G. Interpretation

**HeightScan: Base → Original HeightScan.** Within this preregistered Final realization:

Flat: endpoint reach 97→98/100; completed-section mean traversal 1.052→1.017 s (-0.035 s); Ridge: endpoint reach 96→97/100; completed-section mean traversal 1.457→1.353 s (-0.104 s); Terraces: endpoint reach 93→97/100; completed-section mean traversal 2.232→1.904 s (-0.328 s); Basin: endpoint reach 93→96/100; completed-section mean traversal 1.703→1.582 s (-0.122 s); Landing: endpoint reach 92→94/100; completed-section mean traversal 3.754→3.372 s (-0.383 s).
Fall count before first course end: 8→6; fall after course completion: 11→17; total original falls: 19→23.

**Contact fusion: Original HeightScan → HeightScan+Contact.** Within this preregistered Final realization:

Flat: endpoint reach 98→96/100; completed-section mean traversal 1.017→1.081 s (+0.064 s); Ridge: endpoint reach 97→96/100; completed-section mean traversal 1.353→1.416 s (+0.063 s); Terraces: endpoint reach 97→95/100; completed-section mean traversal 1.904→1.864 s (-0.040 s); Basin: endpoint reach 96→94/100; completed-section mean traversal 1.582→1.599 s (+0.018 s); Landing: endpoint reach 94→94/100; completed-section mean traversal 3.372→3.472 s (+0.101 s).
Fall count before first course end: 6→6; fall after course completion: 17→10; total original falls: 23→16.

**Diverse training: HeightScan+Contact → Diverse HeightScan+Contact.** Within this preregistered Final realization:

Flat: endpoint reach 96→97/100; completed-section mean traversal 1.081→0.996 s (-0.085 s); Ridge: endpoint reach 96→95/100; completed-section mean traversal 1.416→1.344 s (-0.072 s); Terraces: endpoint reach 95→93/100; completed-section mean traversal 1.864→1.751 s (-0.112 s); Basin: endpoint reach 94→91/100; completed-section mean traversal 1.599→1.555 s (-0.044 s); Landing: endpoint reach 94→85/100; completed-section mean traversal 3.472→3.367 s (-0.105 s).
Fall count before first course end: 6→15; fall after course completion: 10→18; total original falls: 16→33.

These comparisons are descriptive; no policy ranking or aggregate score is computed. Time estimates condition on different survivor sets. The section analysis suggests where progress and failures occur, but it does not establish a general causal effect across unseen terrains. CIs are marginal within-realization intervals; no claim of significant between-policy section differences is made.

### First course versus post-course movement

| Policy | Reach local 40 | Original mean displacement m | Mean terminal distance beyond local 40 among reachers m | Mean movement since sampled crossing m | Unreached: mean legacy final displacement m |
|---|---:|---:|---:|---:|---:|
| Base | 92/100 | 55.772 | 22.009 | 21.973 | 7.820 |
| Original HeightScan | 94/100 | 62.437 | 27.528 | 27.491 | 15.098 |
| HeightScan+Contact | 94/100 | 61.914 | 27.606 | 27.570 | 5.160 |
| Diverse HeightScan+Contact | 85/100 | 59.295 | 28.586 | 28.547 | 18.409 |

The 40 m analytic tile starts 2 m behind the robot. First-course end therefore requires about 38 m forward displacement, not 40 m displacement. Original 55–62 m mean displacement is a whole-first-episode metric that includes movement beyond the initial tile. It must not be read as first-course completion or section performance. The first-pass tables stop at the initial local x=40 boundary; subsequent x progress is separated above. Adjacent repeated tiles and outer padding cannot be apportioned from world-X alone, and no claim of completing a bounded 2D corridor is inferred from an X threshold.

## Integrity and provenance

All 468 protected file/checkpoint hashes remained unchanged. No training, tuning, selection, geometry, seed, reward, termination, observation or sensor changes. Existing 400-row Final CSV, summary, preregistration, parity results and prior addendum were preserved.

Trajectory/consistency manifest: `final_unseen_first_pass_trajectory_manifest.json`; episode data: `final_unseen_first_pass_raw.csv`; summary CSV/JSON: `final_unseen_section_analysis.*`; separate addendum: `final_unseen_first_pass_analysis_addendum.json`. Full runtime commands, original hash snapshot, per-policy terminal-safe telemetry and validation logs are under `final_unseen_first_pass_runs/`.

![First-pass completion with Wilson intervals](final_unseen_section_analysis.png)
