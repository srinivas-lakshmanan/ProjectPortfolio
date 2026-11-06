from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from dagger_mpc_policy import collect_episode, episode_interval, write_aggregate
from train_mpc_policy import INPUT_NAMES, STATE_NAMES, fixed_episode_split, read_dataset
from test.test_train_mpc_policy import write_fixture


class DaggerTests(unittest.TestCase):
    def test_learner_drives_plant_and_expert_labels_visited_states(self) -> None:
        class Learner:
            def predict(self, sample):
                return dict.fromkeys(INPUT_NAMES, 0.0)

        class Expert:
            def __init__(self):
                self.states = []
                self.previous = []
                self.resets = 0

            def reset(self):
                self.resets += 1

            def solve(self, sample):
                values = sample["values"]
                self.states.append([values[f"state_{name}"] for name in STATE_NAMES])
                self.previous.append([values[f"previous_{name}"] for name in INPUT_NAMES])
                return {f"expert_{name}": 0.12 if name == "delta" else 180.0
                        for name in INPUT_NAMES}

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.csv"
            write_fixture(path)
            samples, _ = read_dataset(path)
            source = samples[0]
            source["values"]["state_vx"] = 6.0
            for name in STATE_NAMES[6:]:
                source["values"][f"state_{name}"] = 6.0 / 0.31
            second = {**source, "values": source["values"].copy()}
            second["values"]["time_s"] = 0.05
            expert = Expert()
            rows = collect_episode([source, second], Learner(), expert, 0.05, 50, 1)
            self.assertEqual(expert.resets, 1)
            self.assertNotEqual(expert.states[0][3], expert.states[1][3])
            self.assertEqual(expert.previous[1], [0.0] * 5)
            self.assertEqual(rows[1]["values"]["applied_delta"], 0.0)
            self.assertEqual(rows[1]["values"]["expert_delta"], 0.12)
            self.assertEqual(rows[1]["values"]["state_vy"], 0.0)
            self.assertEqual(source["values"]["state_vx"], 6.0)
            self.assertEqual(rows[0]["values"]["dagger_iteration"], 1)
            write_aggregate(Path(directory) / "aggregate.csv", rows)
            loaded, _ = read_dataset(Path(directory) / "aggregate.csv")
            self.assertEqual(loaded[1]["values"]["expert_delta"], 0.12)

    def test_augmented_rows_remain_training_only(self) -> None:
        samples = [{"episode_id": "train"}, {"episode_id": "validation"},
                   {"episode_id": "test"}, {"episode_id": "train"}]
        groups = {"train": ["train"], "validation": ["validation"], "test": ["test"]}
        indices = fixed_episode_split(samples, groups)
        self.assertEqual(indices, {"train": [0, 3], "validation": [1], "test": [2]})
        with self.assertRaisesRegex(ValueError, "exactly once"):
            fixed_episode_split(samples, {**groups, "test": ["train"]})

    def test_nonuniform_episode_is_rejected(self) -> None:
        samples = [{"values": {"time_s": time}} for time in (0.0, 0.05, 0.13)]
        with self.assertRaisesRegex(ValueError, "uniformly increasing"):
            episode_interval(samples)


if __name__ == "__main__":
    unittest.main()
