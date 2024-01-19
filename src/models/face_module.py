import os.path as osp
import logging
from typing import Any, Dict, Tuple
from icecream import ic

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy, BinaryAccuracy
from torchmetrics.classification.f_beta import FBetaScore, BinaryFBetaScore
from torchmetrics.classification.auroc import AUROC, BinaryAUROC
from .components.utils import load_pretrained

logger = logging.getLogger(__name__)


class FaceLitModule(LightningModule):
    """`LightningModule` for Face Color classification."""

    def __init__(
        self,
        net: torch.nn.Module,
        pretrained: str,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        num_heads: int,
        compile: bool,
        num_classes: Tuple[int, int, int, int, int, int],
        attributes: Tuple[str, str, str, str, str, str],
        binary_threshold: float = 0.5,
        binary_beta: float = 1.0,
    ) -> None:
        """Initialize a `FaceLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        if isinstance(pretrained, str) and osp.exists(pretrained):
            logger.info(f"Loading weights from {pretrained}.")
            pretrained_state_dict = torch.load(pretrained)
            state_dict = self.net.state_dict()
            load_pretrained(state_dict, pretrained_state_dict)

        self.dynamic_weights_loss = None
        if hasattr(self.net, "dynamic_weights_loss"):
            self.dynamic_weights_loss = self.net.dynamic_weights_loss

        self.cur_val = 0

        # loss function
        # metric objects for calculating and averaging accuracy across batches
        for i in range(self.hparams.num_heads):
            attr = attributes[i]
            if attr != "age":
                setattr(
                    self,
                    f"criterion_{attr}",
                    torch.nn.CrossEntropyLoss(reduction=("none" if self.dynamic_weights_loss else "mean")),
                )
                setattr(
                    self,
                    f"train_acc_{attr}",
                    Accuracy(task="multiclass", num_classes=num_classes[i]),
                )
                setattr(
                    self,
                    f"val_acc_{attr}",
                    Accuracy(task="multiclass", num_classes=num_classes[i]),
                )
                setattr(
                    self,
                    f"test_acc_{attr}",
                    Accuracy(task="multiclass", num_classes=num_classes[i]),
                )
                setattr(
                    self,
                    f"test_f1_{attr}",
                    FBetaScore(
                        task="multiclass",
                        num_classes=num_classes[i],
                    ),
                )
                setattr(
                    self,
                    f"test_auroc_{attr}",
                    AUROC(task="multiclass", num_classes=num_classes[i]),
                )
            else:
                setattr(self, f"criterion_{attr}", torch.nn.BCELoss(reduction=("none" if self.dynamic_weights_loss else "mean")))
                setattr(
                    self,
                    f"train_acc_{attr}",
                    BinaryAccuracy(threshold=binary_threshold),
                )
                setattr(
                    self,
                    f"val_acc_{attr}",
                    BinaryAccuracy(threshold=binary_threshold),
                )
                setattr(
                    self,
                    f"test_acc_{attr}",
                    BinaryAccuracy(threshold=binary_threshold),
                )
                setattr(
                    self,
                    f"test_f1_{attr}",
                    BinaryFBetaScore(beta=binary_beta),
                )
                setattr(self, f"test_auroc_{attr}", BinaryAUROC())

            # for averaging loss across batches
            setattr(self, f"train_loss_{attr}", MeanMetric())
            setattr(self, f"val_loss_{attr}", MeanMetric())
            setattr(self, f"test_loss_{attr}", MeanMetric())

        setattr(self, "val_acc_all", MeanMetric())
        setattr(self, "val_acc", MeanMetric())
        setattr(self, "val_loss", MeanMetric())

        # for tracking best so far validation accuracy
        setattr(self, "val_acc_best", MaxMetric())

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
            attr = self.hparams.attributes[i]
            getattr(self, f"val_acc_{attr}").reset()
            getattr(self, f"val_loss_{attr}").reset()

        getattr(self, "val_acc").reset()
        getattr(self, "val_loss").reset()
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
        xs, ys = batch
        # breakpoint()
        preds = self.forward(xs)
        preds_dict = {}
        losses = {}

        for i in range(self.hparams.num_heads):
            attr = self.hparams.attributes[i]
            if attr == "age":
                preds_dict[attr] = preds[i].float().squeeze()
                ys[attr] = ys[attr].float()

            preds_dict[attr] = preds[i].float().squeeze()

            loss = getattr(self, f"criterion_{attr}")(preds_dict[attr], ys[attr])
            losses[attr] = loss

        return losses, preds_dict, ys

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
        for i in range(self.hparams.num_heads):
            attr = self.hparams.attributes[i]
            getattr(self, f"train_loss_{attr}")(losses[attr])
            getattr(self, f"train_acc_{attr}")(preds[attr], targets[attr])

            self.log(
                f"train/loss_{attr}",
                getattr(self, f"train_loss_{attr}"),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
            self.log(
                f"train/acc_{attr}",
                getattr(self, f"train_acc_{attr}"),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        # return loss or backpropagation will fail
        losses = torch.stack(list(losses.values()))
        if self.hparams.num_heads > 1:
            if self.dynamic_weights_loss:
                # ic(losses.shape, preds["loss_weights"].shape)
                dynamic_weights_loss = (preds["loss_weights"] / losses).sum()
                loss_final = (losses * preds["loss_weights"]).sum() + dynamic_weights_loss
            else:
                loss_final = losses.mean()
        return loss_final

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
        losses, preds, targets = self.model_step(batch)
        for i in range(self.hparams.num_heads):
            attr = self.hparams.attributes[i]
            val_loss = getattr(self, f"val_loss_{attr}")(
                losses[attr]
            )
            val_acc = getattr(self, f"val_acc_{attr}")(preds[attr], targets[attr])
            getattr(self, "val_acc_all")(val_acc)
            getattr(self, "val_loss")(val_loss)
            self.log(
                f"val/loss_{attr}",
                getattr(self, f"val_loss_{attr}"),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
            self.log(
                f"val/acc_{attr}",
                getattr(self, f"val_acc_{attr}"),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        val_acc_all = getattr(self, "val_acc_all").compute()
        getattr(self, "val_acc")(val_acc_all)

        self.log(
            "val/acc_all",
            val_acc_all,
            sync_dist=True,
            prog_bar=True,
        )
        self.log(
            "val/loss",
            getattr(self, "val_loss").compute(),
            sync_dist=True,
            prog_bar=True,
        )

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
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
        losses, preds, targets = self.model_step(batch)

        for i in range(self.hparams.num_heads):
            attr = self.hparams.attributes[i]
            pred = preds[attr]
            gt = targets[attr]

            getattr(self, f"test_acc_{attr}")(pred, gt)
            getattr(self, f"test_f1_{attr}")(pred, gt)

            self.log(
                f"test/acc_{attr}",
                getattr(self, f"test_acc_{attr}"),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
            self.log(
                f"test/f1_{attr}",
                getattr(self, f"test_f1_{attr}"),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

            if self.hparams.attributes[i] != "age":
                getattr(self, f"test_auroc_{attr}")(pred, gt.long())
                self.log(
                    f"test/auroc_{attr}",
                    getattr(self, f"test_auroc_{attr}"),
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
        optimizer = self.hparams.optimizer(
            params=self.trainer.model.parameters()
        )
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
