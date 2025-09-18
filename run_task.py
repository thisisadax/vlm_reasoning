import hydra
from omegaconf import DictConfig
import pyrootutils

# project root setup
root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)

@hydra.main(version_base=None, config_path="config", config_name="run")
def main(cfg: DictConfig) -> None:
    
    # Instantiate the task from config
    task = hydra.utils.instantiate(cfg.task)
    task.run()
    print(f"✅ Task '{task.task_name}' loaded successfully!")
    
    # Run inference with the model.
    model = hydra.utils.instantiate(cfg.model, task=task)
    model.run()
    

if __name__ == "__main__":
    main()