from data import build_data
from model import build_model
from configs import build_config

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

if __name__ == "__main__":
    seed_everything(0, workers=True)
    cfg = build_config()
    dataset = build_data(cfg)
    model = build_model(cfg)
    # checkpoint_callback = ModelCheckpoint(
    #     save_top_k=10,
    #     monitor= "EncodedAssistQADataModule loss",
    #     mode='min'
    # )
    checkpoint_callback = ModelCheckpoint(
        save_top_k=10,
        monitor= "EncodedAssistQADataModule recall@1",
        mode='max',
    )
    trainer = Trainer(
        devices=cfg.NUM_GPUS, 
        accelerator="gpu",
        strategy=DDPStrategy(find_unused_parameters=False),
        callbacks=[
            LearningRateMonitor(logging_interval='step'), 
            checkpoint_callback
        ],
        benchmark=False, 
        deterministic=True,
        max_epochs=cfg.SOLVER.MAX_EPOCHS,
        default_root_dir=cfg.OUTPUT_DIR,
        check_val_every_n_epoch=cfg.CHECK_VAL_EVERY_N_EPOCH,
        num_sanity_val_steps=0,
        log_every_n_steps=5,
        
    )
    trainer.fit(model, datamodule=dataset, 
        ckpt_path=cfg.CKPT if hasattr(cfg, "CKPT") else None)
    