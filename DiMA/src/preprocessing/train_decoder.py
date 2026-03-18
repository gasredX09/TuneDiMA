import os
import wandb
import argparse
import torch
from tqdm import tqdm
from contextlib import nullcontext
from hydra.utils import instantiate

from src.utils.hydra_utils import setup_config
from src.utils import seed_everything
from src.preprocessing.preprocessing_utils import get_loaders
from src.utils.training_utils import reconstruction_loss
from src.utils.logging_utils import config_to_wandb


def save_decoder_checkpoint(path, decoder, optimizer, step):
    state_dict = {
        "decoder": decoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
    }
    torch.save(state_dict, path)


def loss_step(batch, encoder, config, dynamic, amp_ctx, eval=False):
    # Encoder is frozen during decoder training; avoid building autograd graph through ESM2.
    with torch.no_grad():
        latent, attention_mask, input_ids = encoder.batch_encode(
            batch,
            max_sequence_len=config.datasets.max_sequence_len,
        )
    attention_mask = attention_mask.int()

    if not eval:
        T = config.decoder.max_T
        eps = config.decoder.min_T
        t = torch.cuda.FloatTensor(latent.shape[0]).uniform_() * (T - eps) + eps
        x_t = dynamic.marginal(latent, t)["x_t"]
        latent = x_t
    
    latent = encoder.enc_normalizer.denormalize(latent)

    with amp_ctx():
        logits = encoder.batch_get_logits(latent, attention_mask)
    loss = reconstruction_loss(input_ids, logits, mask=attention_mask)
    
    tokens = logits.argmax(dim=-1)
    acc = torch.mean((input_ids == tokens) * 1.)

    return loss, acc


def train_decoder(config, encoder, train_loader, valid_loader):
    decoder = encoder.sequence_decoder
    total_number_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f"Num params: {total_number_params}")

    scheduler = instantiate(config.scheduler)
    dynamic = instantiate(config.dynamic, scheduler=scheduler)

    device = encoder.device
    if device.type == "cuda":
        bf16_supported = hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16 if bf16_supported else torch.float16
        amp_ctx = lambda: torch.autocast(device_type='cuda', dtype=amp_dtype)
    else:
        amp_ctx = nullcontext

    optimizer = instantiate(config.decoder.optimizer, params=decoder.parameters())

    decoder_checkpoint_dir = os.getenv("DECODER_CHECKPOINT_DIR")
    decoder_ckpt_interval = int(os.getenv("DECODER_CKPT_INTERVAL", "0"))
    decoder_max_steps = int(os.getenv("DECODER_MAX_STEPS", "0"))
    decoder_resume_path = os.getenv("DECODER_RESUME_PATH")

    if decoder_checkpoint_dir:
        os.makedirs(decoder_checkpoint_dir, exist_ok=True)

    step = 0
    if decoder_resume_path and os.path.exists(decoder_resume_path):
        ckpt = torch.load(decoder_resume_path, map_location="cpu")
        if "decoder" in ckpt:
            decoder.load_state_dict(ckpt["decoder"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        step = int(ckpt.get("step", 0))
        print(f"Resumed decoder from {decoder_resume_path} at step={step}")

    for _ in range(config.decoder.training_epochs):
        decoder.train()

        for batch in tqdm(train_loader):
            if decoder_max_steps > 0 and step >= decoder_max_steps:
                break

            loss, acc = loss_step(
                batch=batch,
                encoder=encoder,
                config=config,
                dynamic=dynamic,
                amp_ctx=amp_ctx,
            )
       
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                decoder.parameters(),
                max_norm=1.0
            )
            optimizer.step()

            if wandb.run is not None:
                wandb.log({f'train loss': loss.item()}, step=step)
                wandb.log({f'train accuracy': acc.item()}, step=step)

            step += 1

            if decoder_checkpoint_dir and decoder_ckpt_interval > 0 and step % decoder_ckpt_interval == 0:
                ckpt_path = os.path.join(decoder_checkpoint_dir, f"decoder_step_{step}.pth")
                save_decoder_checkpoint(ckpt_path, decoder, optimizer, step)
                last_path = os.path.join(decoder_checkpoint_dir, "decoder_last.pth")
                save_decoder_checkpoint(last_path, decoder, optimizer, step)

        if decoder_max_steps > 0 and step >= decoder_max_steps:
            print(f"Reached DECODER_MAX_STEPS={decoder_max_steps}. Stopping training loop.")
            break
            
        decoder.eval()
        total_loss = 0
        total_acc = 0
        total_samples = 0

        for batch in tqdm(valid_loader):
            with torch.no_grad():
                loss, acc = loss_step(
                    batch=batch,
                    encoder=encoder,
                    config=config,
                    dynamic=dynamic,
                    amp_ctx=amp_ctx,
                    eval=True
                )

                batch_size = len(batch["sequence"])
                total_loss += loss.item() * batch_size
                total_acc += acc.item() * batch_size
                total_samples += batch_size

        if wandb.run is not None:
            wandb.log({f'valid loss': total_loss / total_samples}, step=step)
            wandb.log({f'valid accuracy': total_acc / total_samples}, step=step)

    return decoder


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--project_path", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    args = parser.parse_args()

    config = setup_config(config_path=args.config_path)

    if args.project_path is not None:
        config.project.path = args.project_path

    if args.data_dir is not None:
        config.datasets.data_dir = args.data_dir

    decoder_num_workers = os.getenv("DECODER_NUM_WORKERS")
    if decoder_num_workers is not None:
        config.dataloader.num_workers = int(decoder_num_workers)

    decoder_batch_size = os.getenv("DECODER_BATCH_SIZE_PER_GPU")
    if decoder_batch_size is not None:
        config.training.batch_size_per_gpu = int(decoder_batch_size)
        config.training.batch_size = int(decoder_batch_size)

    decoder_max_sequence_len = os.getenv("DECODER_MAX_SEQUENCE_LEN")
    if decoder_max_sequence_len is not None:
        config.datasets.max_sequence_len = int(decoder_max_sequence_len)

    seed_everything(config.project.seed)

    disable_wandb = os.getenv("DISABLE_WANDB", "0") in {"1", "true", "TRUE", "yes", "YES"}
    use_wandb = (not disable_wandb) and (not config.ddp.enabled or config.ddp.global_rank == 0)

    # ✅ Initialize Weights and Biases
    if use_wandb:
        name = config.project.checkpoints_prefix
        wandb_mode = os.getenv("WANDB_MODE", "online")
        wandb.init(
            project=config.project.wandb_project,
            name=name,
            mode=wandb_mode
        )
        config_to_wandb(config)

    device = torch.device("cuda:0")

    encoder_partial = instantiate(config.encoder)
    encoder = encoder_partial(
        device=device,
        main_config=config,
        add_enc_normalizer=True,
    )

    # Train decoder
    train_loader, valid_loader = get_loaders(config=config)
    decoder = train_decoder(
        config=config,
        encoder=encoder,
        train_loader=train_loader,
        valid_loader=valid_loader,
    )
    
    # Save statistics
    os.makedirs(config.project.decoder_checkpoints_folder, exist_ok=True)
    decoder.eval()

    state_dict = {
        "decoder": decoder.state_dict(),
    }
    
    path = config.decoder.decoder_path
    torch.save(state_dict, path)
    print(f"Save preprocessing to: {path}")
