import wandb


class WandBWrapper:
    def __init__(self, config, resume=False, resume_id=None):
        self.config = config
        wandb.init(project="Not LegUp", config=config, entity="mlvovsky",
                   sync_tensorboard=True, monitor_gym=True, save_code=True, resume=resume, id=resume_id)

        self.id = wandb.run.id

    def recover(self):
        wandb.init(project="Not LegUp", config=self.config, entity="mlvovsky",
                   sync_tensorboard=True, monitor_gym=True, save_code=True, resume=True, id=self.wandb_id)

    def finish(self):
        wandb.finish()

    def is_initialized(self):
        return wandb.run is not None

    def restore(self, path):
        return wandb.restore(path)

    def is_resumed(self):
        return wandb.run.resumed
