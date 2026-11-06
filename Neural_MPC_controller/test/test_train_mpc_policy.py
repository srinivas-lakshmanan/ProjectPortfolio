import csv
import importlib.util
import io
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_mpc_policy.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("train_mpc_policy", SCRIPT)
policy = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(policy)
import rollout_mpc_policy as rollout


def write_fixture(path: Path, episodes: int = 3) -> None:
    horizon = 2
    headers = [
        "episode_id", "maneuver", "time_s", "speed_m_s", "step_angle_rad",
        "lane_width_m", "initial_y_error_m",
    ]
    headers += [f"state_{name}" for name in policy.STATE_NAMES]
    headers += [f"previous_{name}" for name in policy.INPUT_NAMES]
    headers += [
        f"reference_{step}_{name}"
        for step in range(horizon + 1) for name in policy.STATE_NAMES
    ]
    headers += [
        f"reference_input_{step}_{name}"
        for step in range(horizon) for name in policy.INPUT_NAMES
    ]
    headers += list(policy.OUTPUT_NAMES)

    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=headers)
        writer.writeheader()
        for episode in range(episodes):
            row = {name: "0" for name in headers}
            row.update(episode_id=str(episode), maneuver="step-steer", speed_m_s="8")
            for step in range(horizon + 1):
                row[f"reference_{step}_X"] = str(step * 0.4)
                row[f"reference_{step}_vx"] = "8"
            for step in range(horizon):
                row[f"reference_input_{step}_delta"] = "0.01"
            row["expert_delta"] = str(0.01 * episode)
            writer.writerow(row)


class TrainMpcPolicyTests(unittest.TestCase):
    def test_feature_transform_is_translation_invariant(self) -> None:
        sample = {"values": {}}
        for name in policy.STATE_NAMES:
            sample["values"][f"state_{name}"] = 0.0
        for name in policy.INPUT_NAMES:
            sample["values"][f"previous_{name}"] = 0.0
        for step in range(3):
            for name in policy.STATE_NAMES:
                sample["values"][f"reference_{step}_{name}"] = 0.0
            sample["values"][f"reference_input_{step // 2}_{policy.INPUT_NAMES[0]}"] = 0.0
        for step in range(2):
            for name in policy.INPUT_NAMES:
                sample["values"][f"reference_input_{step}_{name}"] = 0.0
        sample["values"].update({
            "state_X": 10.0, "state_Y": -3.0, "state_psi": math.pi / 2,
            "reference_0_X": 11.0, "reference_0_Y": -3.0,
            "reference_0_psi": math.pi / 2,
        })
        original = policy.make_features(sample, 2)
        sample["values"]["state_X"] += 125.0
        sample["values"]["state_Y"] -= 78.0
        for step in range(3):
            sample["values"][f"reference_{step}_X"] += 125.0
            sample["values"][f"reference_{step}_Y"] -= 78.0
        moved = policy.make_features(sample, 2)
        self.assertEqual(original, moved)
        self.assertAlmostEqual(original[12], 0.0, places=12)
        self.assertAlmostEqual(original[13], -1.0, places=12)

    def test_csv_schema_and_episode_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.csv"
            write_fixture(path)
            samples, horizon = policy.read_dataset(path)
            self.assertEqual(horizon, 2)
            self.assertEqual(len(samples), 3)
            self.assertEqual(
                len(policy.make_features(samples[0], horizon)),
                len(policy.feature_names(horizon)),
            )
            indices, episodes = policy.split_by_episode(samples, 0.15, 0.15, 4)
            self.assertEqual(set(indices), {"train", "validation", "test"})
            memberships = [set(values) for values in episodes.values()]
            self.assertFalse(memberships[0] & memberships[1])
            self.assertFalse(memberships[0] & memberships[2])
            self.assertFalse(memberships[1] & memberships[2])

    def test_nonfinite_data_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.csv"
            write_fixture(path)
            with path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            rows[0]["state_vx"] = "nan"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "not finite"):
                policy.read_dataset(path)

    def test_policy_checkpoint_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "policy.pt"
            horizon = 2
            input_size = len(policy.feature_names(horizon))
            model = policy.build_model(input_size, [4], len(policy.INPUT_NAMES))
            checkpoint = {
                "format_version": 1,
                "model_state_dict": model.state_dict(),
                "input_size": input_size,
                "hidden_sizes": [4],
                "output_names": list(policy.INPUT_NAMES),
                "feature_names": policy.feature_names(horizon),
                "horizon": horizon,
                "normalization": {
                    "feature_mean": [0.0] * input_size,
                    "feature_scale": [1.0] * input_size,
                    "target_mean": [0.0] * len(policy.INPUT_NAMES),
                    "target_scale": [1.0] * len(policy.INPUT_NAMES),
                },
            }
            torch = policy.torch
            torch.save(checkpoint, checkpoint_path)
            predictor = policy.MpcPolicy(checkpoint_path)
            sample = {"values": {}}
            for name in policy.STATE_NAMES:
                sample["values"][f"state_{name}"] = 0.0
            for name in policy.INPUT_NAMES:
                sample["values"][f"previous_{name}"] = 0.0
            for step in range(horizon + 1):
                for name in policy.STATE_NAMES:
                    sample["values"][f"reference_{step}_{name}"] = 0.0
            for step in range(horizon):
                for name in policy.INPUT_NAMES:
                    sample["values"][f"reference_input_{step}_{name}"] = 0.0
            prediction = predictor.predict(sample)
            self.assertEqual(set(prediction), set(policy.INPUT_NAMES))
            self.assertTrue(all(math.isfinite(value) for value in prediction.values()))

    def test_control_projection_enforces_bounds_and_slew(self) -> None:
        args = SimpleNamespace(
            steering_limit=0.15,
            steering_rate_limit=0.30,
            torque_limit=200.0,
            torque_rate_limit=1500.0,
        )
        applied = rollout.constrain_control(
            [1.0, 500.0, -500.0, 50.0, -50.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            0.05,
            args,
        )
        self.assertEqual(applied, [0.015, 75.0, -75.0, 50.0, -50.0])
        self.assertLessEqual(abs(applied[0]), args.steering_limit)
        self.assertLessEqual(abs(applied[1]), args.torque_limit)

    def test_rollout_feeds_simulated_state_back_to_policy(self) -> None:
        class RecordingPolicy:
            horizon = 2

            def __init__(self) -> None:
                self.longitudinal_speeds = []

            def predict(self, sample):
                self.longitudinal_speeds.append(sample["values"]["state_vx"])
                return dict.fromkeys(policy.INPUT_NAMES, 0.0)

        samples = []
        for step in range(2):
            values = {"time_s": step * 0.05}
            for name in policy.STATE_NAMES:
                values[f"state_{name}"] = 0.0
            values.update(state_vx=6.0)
            for name in policy.INPUT_NAMES:
                values[f"previous_{name}"] = 0.0
            for preview in range(3):
                for name in policy.STATE_NAMES:
                    values[f"reference_{preview}_{name}"] = 0.0
                values[f"reference_{preview}_vx"] = 6.0
            for preview in range(2):
                for name in policy.INPUT_NAMES:
                    values[f"reference_input_{preview}_{name}"] = 0.0
            samples.append({
                "episode_id": "1", "maneuver": "step-steer", "values": values,
            })
        samples[0]["values"].update({
            "state_w_fl": 6.0 / 0.31,
            "state_w_fr": 6.0 / 0.31,
            "state_w_rl": 6.0 / 0.31,
            "state_w_rr": 6.0 / 0.31,
        })
        samples[1]["values"].update({
            "state_w_fl": 6.0 / 0.31,
            "state_w_fr": 6.0 / 0.31,
            "state_w_rl": 6.0 / 0.31,
            "state_w_rr": 6.0 / 0.31,
        })
        args = SimpleNamespace(
            rk4_substeps=50,
            steering_limit=0.15,
            steering_rate_limit=0.30,
            torque_limit=200.0,
            torque_rate_limit=1500.0,
        )
        fieldnames = ["episode_id", "maneuver", "time_s"]
        fieldnames += [f"state_{name}" for name in policy.STATE_NAMES]
        fieldnames += [f"reference_{name}" for name in policy.STATE_NAMES]
        fieldnames += [f"requested_{name}" for name in policy.INPUT_NAMES]
        fieldnames += [f"applied_{name}" for name in policy.INPUT_NAMES]
        fieldnames += [f"previous_{name}" for name in policy.INPUT_NAMES]
        fieldnames += ["lateral_error_m", "heading_error_rad", "speed_error_m_s"]
        predictor = RecordingPolicy()
        rollout.rollout(
            samples, "1", predictor, args,
            csv.DictWriter(io.StringIO(), fieldnames=fieldnames),
        )
        self.assertEqual(predictor.longitudinal_speeds[0], 6.0)
        self.assertNotEqual(predictor.longitudinal_speeds[1], 6.0)


if __name__ == "__main__":
    unittest.main()
