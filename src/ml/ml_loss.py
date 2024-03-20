
    
class LossStatistics:
    """
    The class that accumulate the loss function values over all batches
    for each loss component.
    
    """
    
    def __init__(self):
        import pandas as pd
        # conter # of epochs
        # self.iepoch = 0
        self.iepoch_valid_list = []
        self.iepoch_train_list = []

        
        # loss / rmse list for each batch
        self.valid_rmse_list  = []
        self.train_rmse_list = []
        self.valid_loss_list  = []
        self.train_loss_list = []
        # https://bering.hatenadiary.com/entry/2023/05/15/064223
        # validation rmse/loss for each batch
        self.df_batch_valid = [] # pd.DataFrame(["epoch", "batch", "loss", "rmse"])
        # training rmse/loss for each batch
        self.df_batch_train = [] # pd.DataFrame(["epoch", "batch", "loss", "rmse"])
        

        # epoch loss (train and valid)
        # validation/training rmse/loss for each epoch
        self.df_epoch = [] # pd.DataFrame(["epoch", "train_loss", "valid_loss", "train_rmse", "valid_rmse"])
        
        self.epoch_valid_loss_list = []
        self.epoch_train_loss_list = []
        self.epoch_valid_rmse_list = []
        self.epoch_train_rmse_list = []
        
    def add_train_batch_loss(self, loss, iepoch:int) -> None:
        import numpy as np
        # TODO :: rmse is not always np.sqrt(loss).
        # TODO :: 
        self.df_batch_train.append({"epoch":iepoch, "batch":len(self.df_batch_train), "loss":loss.item(), "rmse":np.sqrt(loss.item())})
        
        # self.train_rmse_list.append(np.sqrt(loss.item()))
        # self.train_loss_list.append(loss.item()) 
        # update # of epoch 
        # self.iepoch_train_list.append()

    def add_valid_batch_loss(self, loss, iepoch:int) -> None:
        import numpy as np
        self.df_batch_valid.append({"epoch":iepoch, "batch":len(self.df_batch_valid), "loss":loss.item(), "rmse":np.sqrt(loss.item())})
        
        # self.valid_rmse_list.append(np.sqrt(loss.item()))
        # self.valid_loss_list.append(loss.item()) 
        # update # of epoch 
        # self.iepoch_valid_list.append()


    
    
    def __init__(self, loss_instance=None):
        self.loss_stat = {
            "total": RunningStats(
                dim=tuple(), reduction=Reduction.MEAN, ignore_nan=False
            )
        }
        self.ignore_nan = {}
        if loss_instance is not None:
            for key, func in loss_instance.funcs.items():
                self.ignore_nan[key] = (
                    func.ignore_nan if hasattr(func, "ignore_nan") else False
                )

    def __call__(self, loss, loss_contrib):
        """
        Args:

        loss (torch.Tensor): the value of the total loss function for the current batch
        loss (Dict(torch.Tensor)): the dictionary which contain the loss components
        """

        results = {}

        results["loss"] = self.loss_stat["total"].accumulate_batch(loss).item()

        # go through each component
        for k, v in loss_contrib.items():

            # initialize for the 1st batch
            if k not in self.loss_stat:
                self.loss_stat[k] = RunningStats(
                    dim=tuple(),
                    reduction=Reduction.MEAN,
                    ignore_nan=self.ignore_nan.get(k, False),
                )
                device = v.get_device()
                self.loss_stat[k].to(device="cpu" if device == -1 else device)

            results["loss_" + ABBREV.get(k, k)] = (
                self.loss_stat[k].accumulate_batch(v).item()
            )
        return results

    def reset(self):
        """
        Reset all the counters to zero
        """

        for v in self.loss_stat.values():
            v.reset()

    def to(self, device):
        for v in self.loss_stat.values():
            v.to(device=device)

    def current_result(self):
        results = {
            "loss_" + ABBREV.get(k, k): v.current_result().item()
            for k, v in self.loss_stat.items()
            if k != "total"
        }
        results["loss"] = self.loss_stat["total"].current_result().item()
        return results



    def end_of_batch_log(self, batch_type: str):
        """
        store all the loss/mae of each batch
        """

        mat_str = f"{self.iepoch+1:5d}, {self.ibatch+1:5d}"
        log_str = f"  {self.iepoch+1:5d} {self.ibatch+1:5d}"

        header = "epoch, batch"
        log_header = "# Epoch batch"

        # print and store loss value
        for name, value in self.batch_losses.items():
            mat_str += f", {value:16.5g}"
            header += f", {name}"
            log_str += f" {value:12.3g}"
            log_header += f" {name:>12.12}"

        # append details from metrics
        metrics, skip_keys = self.metrics.flatten_metrics(
            metrics=self.batch_metrics,
            type_names=self.dataset_train.type_mapper.type_names,
        )
        for key, value in metrics.items():

            mat_str += f", {value:16.5g}"
            header += f", {key}"
            if key not in skip_keys:
                log_str += f" {value:12.3g}"
                log_header += f" {key:>12.12}"

        batch_logger = logging.getLogger(self.batch_log[batch_type])

        if self.ibatch == 0:
            self.logger.info("")
            self.logger.info(f"{batch_type}")
            self.logger.info(log_header)
            init_step = -1 if self.report_init_validation else 0
            if (self.iepoch == init_step and batch_type == VALIDATION) or (
                self.iepoch == 0 and batch_type == TRAIN
            ):
                batch_logger.info(header)

        batch_logger.info(mat_str)
        if (self.ibatch + 1) % self.log_batch_freq == 0 or (
            self.ibatch + 1
        ) == self.n_batches:
            self.logger.info(log_str)
