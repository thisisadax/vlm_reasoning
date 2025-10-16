# generate_examples.py
import hydra
from omegaconf import DictConfig
from pathlib import Path

@hydra.main(config_path="config", config_name="run", version_base=None)
def main(cfg: DictConfig):
    # Only instantiate the Task and generate stimuli. No model inference.
    task = hydra.utils.instantiate(cfg.task)
    # Ensure output dir exists (Hydra cwd is the run dir)
    Path(task.output_dir).mkdir(parents=True, exist_ok=True)
    task.run()

    trials_path = getattr(task, "trials_metadata_path", None)
    if trials_path is None:
        print("✅ Generation finished (task has no trials_metadata_path).")
    elif (not trials_path.exists()) or trials_path.stat().st_size == 0:
        print(f"⚠️  No trials found at {trials_path}.")
    else:
        print(f"✅ Wrote trials to {trials_path}")

if __name__ == "__main__":
    main()
