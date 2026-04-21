import hydra
import torch
import wandb
import os
import torch.distributed as dist
from src.diffusion.base_trainer import BaseDiffusionTrainer
from src.utils import seed_everything, setup_ddp, print_config
from src.utils.logging_utils import config_to_wandb


@hydra.main(version_base=None, config_path="src/configs", config_name="config")
def main(config):
    # ✅ DDP (Distributed Data Parallel) setup
    if config.ddp.enabled:
        config.ddp.local_rank, config.ddp.global_rank = setup_ddp()
        config.training.batch_size_per_gpu = config.training.batch_size // dist.get_world_size()
        config.dataloader.batch_size = config.training.batch_size_per_gpu
    
    config.model.config.embedding_size = config.encoder.config.embedding_dim

    if config.ddp.global_rank == 0:
        print_config(config)

    # ✅ Seed everything
    seed = config.project.seed + config.ddp.global_rank
    seed_everything(seed)

    # ✅ Initialize Weights and Biases (optional for offline/eval SLURM jobs)
    disable_wandb = os.getenv("DISABLE_WANDB", "0").lower() in {"1", "true", "yes", "on"}
    wandb_mode_env = os.getenv("WANDB_MODE", "online").lower()
    use_wandb = not disable_wandb and wandb_mode_env not in {"disabled", "offline"}

    if not config.ddp.enabled or config.ddp.global_rank == 0:
        if use_wandb:
            name = config.project.checkpoints_prefix
            wandb.init(
                project=config.project.wandb_project,
                name=name,
                mode="online"
            )
            config_to_wandb(config)
        else:
            print("[INFO] W&B disabled for this run")

    device = torch.device(f"cuda:{config.ddp.local_rank}") if config.ddp.enabled else torch.device("cuda")
    trainer = BaseDiffusionTrainer(config, device)
    trainer.train()

    if config.ddp.global_rank == 0 and use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()