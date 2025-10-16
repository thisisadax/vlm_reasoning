# run_task.py
import hydra
from omegaconf import DictConfig
from pathlib import Path
import pandas as pd


@hydra.main(config_path="config", config_name="run", version_base=None)
def main(cfg: DictConfig):
    # 1) instantiate task and run generation
    task = hydra.utils.instantiate(cfg.task)
    task.run()

    # 2) if no trials, skip model inference cleanly
    trials_path = task.trials_metadata_path
    if (not trials_path.exists()) or trials_path.stat().st_size == 0:
        print(f"⚠️  No trials found at {trials_path}. Skipping model inference.")
        return

    # Allow skipping model inference via flag (used by examples.sh data generation)
    if getattr(cfg, "skip_model", False):
        print("⏭️  Skipping model inference as requested (skip_model=true).")
        return

    # 3) otherwise run model
    model = hydra.utils.instantiate(cfg.model, task=task)
    model.run()  # your model class should save outputs internally


if __name__ == "__main__":
    main()