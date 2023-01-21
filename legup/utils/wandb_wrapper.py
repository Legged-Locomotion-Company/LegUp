import wandb

class WandBWrapper:
    def __init__(self, config):
        self.config = config
        wandb.init(project="LegUp", config=config, entity="legged-locomotion-company",
                   sync_tensorboard=True, monitor_gym=True, save_code=True, resume=False)

        self.id = wandb.run.id

    def recover(self):
        wandb.init(project="LegUp", config=self.config, entity="legged-locomotion-company",
                   sync_tensorboard=True, monitor_gym=True, save_code=True, resume=True, id=self.wandb_id)

    def finish(self):
        wandb.finish()
