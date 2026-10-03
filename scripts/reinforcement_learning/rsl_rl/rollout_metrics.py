"""Pure NumPy window metrics shared by online logging and offline recovery/analysis."""

import numpy as np


def window_metrics(all_rows, initial, dt, start, end, scan_enabled):
    start, end = int(start), int(end)
    rows = all_rows[start:end]
    previous = all_rows[start-1] if start else initial
    mean = lambda key: float(np.mean([r[key] for r in rows]))
    return {"start_step": start+1, "end_step": end, "start_time_s": start*dt, "end_time_s": end*dt,
            "displacement_m": rows[-1]["position_0"]-previous["position_0"],
            "mean_vx": mean("linear_velocity_0"), "start_vx": rows[0]["linear_velocity_0"], "end_vx": rows[-1]["linear_velocity_0"],
            "mean_action_l2": mean("action_l2"), "start_action_l2": rows[0]["action_l2"], "end_action_l2": rows[-1]["action_l2"],
            "mean_contact_count": mean("contact_count"), "start_contact_count": rows[0]["contact_count"], "end_contact_count": rows[-1]["contact_count"],
            "roll_rms_rad": float(np.sqrt(np.mean([r["roll"]**2 for r in rows]))),
            "pitch_rms_rad": float(np.sqrt(np.mean([r["pitch"]**2 for r in rows]))),
            "start_roll_rad": rows[0]["roll"], "end_roll_rad": rows[-1]["roll"],
            "start_pitch_rad": rows[0]["pitch"], "end_pitch_rad": rows[-1]["pitch"],
            "scan_discontinuity_mean_m": mean("scan_adjacent_difference_max_m") if scan_enabled else None}
