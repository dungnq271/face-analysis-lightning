from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy


class FaceLitModule(LightningModule):
    """`LightningModule` for Face Color classification."""

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        num_heads: int,
        compile: bool,
        num_classes: Tuple[int, int, int, int, int, int],
        attributes: Tuple[str, str, str, str, str, str],
    ) -> None:
        """Initialize a `FaceLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore="net")

        self.net = net
        self.attrs = attributes
        # loss function
        # metric objects for calculating and averaging accuracy across batches
        for i in range(self.hparams.num_heads):
            setattr(self, f"criterion_{self.attrs[i]}",
                    torch.nn.CrossEntropyLoss())
            setattr(self,
                    f"train_acc_{self.attrs[i]}",
                    Accuracy(task="multiclass", num_classes=num_classes[i])
                    )
            setattr(self,
                    f"val_acc_{self.attrs[i]}",
                    Accuracy(task="multiclass", num_classes=num_classes[i])
                    )
            setattr(self, f"test_acc_{self.attrs[i]}",
                    Accuracy(task="multiclass", num_classes=num_classes[i])
                    )

            # for averaging loss across batches
            setattr(self, f"train_loss_{self.attrs[i]}", MeanMetric())
            setattr(self, f"val_loss_{self.attrs[i]}", MeanMetric())
            setattr(self, f"test_loss_{self.attrs[i]}", MeanMetric())

            # for tracking best so far validation accuracy
            # setattr(self, f"val_acc_best_{self.attrs[i]}", MaxMetric())
        setattr(self, "val_acc", MeanMetric())
        setattr(self, "val_loss", MeanMetric())
        setattr(self, "val_acc_best", MaxMetric())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        for i in range(self.hparams.num_heads):
            getattr(self, f"val_acc_{self.attrs[i]}").reset()
            getattr(self, f"val_loss_{self.attrs[i]}").reset()
        getattr(self, "val_acc_best").reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        x, y = batch
        # breakpoint()
        pred = self.forward(x)
        losses = {}
        for i in range(self.hparams.num_heads):
            loss = getattr(self, f"criterion_{self.attrs[i]}")(pred[self.attrs[i]],
                                                               y[self.attrs[i]])
            getattr(self, f"train_loss_{self.attrs[i]}")(loss)
            getattr(self, f"train_acc_{self.attrs[i]}")(pred[self.attrs[i]],
                                                        y[self.attrs[i]])
            losses[self.attrs[i]] = loss
        return losses, pred, y

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        losses, preds, targets = self.model_step(batch)
        # update and log metrics
        # self.train_loss(losses)
        # self.train_acc(preds, targets)
        # self.log(
        #     "train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True
        # )
        # self.log(
        #     "train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True
        # )
        for i in range(self.hparams.num_heads):
            getattr(self, f"train_loss_{self.attrs[i]}")(losses[self.attrs[i]])
            getattr(self, f"train_acc_{self.attrs[i]}")(preds[self.attrs[i]],
                                                        targets[self.attrs[i]])
            self.log(
                f"train/loss_{self.attrs[i]}",
                getattr(self, f"train_loss_{self.attrs[i]}"),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                )
            self.log(
                f"train/acc_{self.attrs[i]}",
                getattr(self, f"train_acc_{self.attrs[i]}"),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                )
        # return loss or backpropagation will fail
        if self.hparams.num_heads > 1:
            losses = torch.stack(list(losses.values())).mean()
        return losses

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        # self.val_loss(loss)
        # self.val_acc(preds, targets)
        # self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        val_loss_all = 0
        val_acc_all = 0
        for i in range(self.hparams.num_heads):
            val_loss = getattr(self, f"val_loss_{self.attrs[i]}")(loss[self.attrs[i]])
            val_acc = getattr(self, f"val_acc_{self.attrs[i]}")(preds[self.attrs[i]],
                                                                targets[self.attrs[i]])
            val_loss_all += val_loss
            val_acc_all += val_acc
            self.log(
                f"val/loss_{self.attrs[i]}",
                getattr(self, f"val_loss_{self.attrs[i]}"),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
            self.log(
                f"val/acc_{self.attrs[i]}",
                getattr(self, f"val_acc_{self.attrs[i]}"),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
        val_loss_all /= self.hparams.num_heads
        val_acc_all /= self.hparams.num_heads
        getattr(self, "val_acc")(val_acc_all)
        self.log(
            "val/acc",
            getattr(self, "val_acc").compute(),
            sync_dist=True,
            prog_bar=True,
        )
        getattr(self, "val_loss")(val_loss_all)
        self.log(
            "val/loss",
            getattr(self, "val_loss").compute(),
            sync_dist=True,
            prog_bar=True,
        )
        
    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        # acc = self.val_acc.compute()  # get current val acc
        # self.val_acc_best(acc)  # update best so far val acc
        # # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # # otherwise metric would be reset by lightning after each epoch
        # self.log(
        #     "val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True
        # )
        acc_best = getattr(self, "val_acc").compute()
        getattr(self, "val_acc_best")(acc_best)
        self.log(
            "val/acc_best",
            getattr(self, "val_acc_best").compute(),
            sync_dist=True,
            prog_bar=True,
        )

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        # self.test_loss(loss)
        # self.test_acc(preds, targets)
        # self.log(
        #     "test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True
        # )
        # self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        for i in range(self.hparams.num_heads):
            getattr(self, f"test_loss_{self.attrs[i]}")(loss[self.attrs[i]])
            getattr(self, f"test_acc_{self.attrs[i]}")(preds[self.attrs[i]],
                                                       targets[self.attrs[i]])
            self.log(
                f"test/loss_{self.attrs[i]}",
                getattr(self, f"test_loss_{self.attrs[i]}"),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
            self.log(
                f"test/acc_{self.attrs[i]}",
                getattr(self, f"test_acc_{self.attrs[i]}"),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = FaceLitModule(None, None, None, None)
