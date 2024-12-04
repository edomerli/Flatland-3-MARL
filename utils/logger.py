import wandb

class WandbLogger:
    def __init__(self):
        self.metrics_dict = {}

    def record(self, metric_name, metric_value, *args, **kwargs):
        self.metrics_dict[metric_name] = metric_value
        
    def dump(self, step):
        self.metrics_dict["step"] = step
        wandb.log(self.metrics_dict)
        self.metrics_dict = {}