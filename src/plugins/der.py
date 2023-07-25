import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from avalanche.benchmarks.utils import (classification_subset,
                                        make_tensor_classification_dataset)
from avalanche.models.utils import avalanche_forward
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.training.storage_policy import (BalancedExemplarsBuffer,
                                               ReservoirSamplingBuffer)
from toolkit.utils import cross_entropy_with_soft_targets
from torchvision.transforms import RandomCrop, RandomHorizontalFlip, Compose
import copy
import numpy as np

from avalanche.benchmarks import OnlineCLExperience


def cycle(loader):
    while True:
        for batch in loader:
            yield batch


@torch.no_grad()
def dataset_with_logits(dataset, model, batch_size, device, num_classes):
    model = copy.deepcopy(model)
    model.train()
    logits = []
    data = []
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for x, _, _ in loader:
        x = x.to(device)
        out = model(x)
        logits.append(out)
        data.append(x)

    logits = torch.cat(logits)
    data = torch.cat(data)

    logits = F.pad(logits, (0, 100 - logits.shape[1]), value=0)

    transform = Compose([RandomCrop(data[0].shape[2], padding=4), RandomHorizontalFlip()])

    dataset = make_tensor_classification_dataset(
        data,
        torch.tensor(dataset.targets),
        torch.tensor(dataset.targets_task_labels),
        logits,
        transform=transform,
    )
    return dataset


class ClassBalancedBuffer(BalancedExemplarsBuffer):
    """
    ClassBalancedBuffer that also stores the logits
    """

    def __init__(
        self,
        max_size: int,
        adaptive_size: bool = True,
        total_num_classes: int = None,
    ):
        """Init.

        :param max_size: The max capacity of the replay memory.
        :param adaptive_size: True if mem_size is divided equally over all
                            observed experiences (keys in replay_mem).
        :param total_num_classes: If adaptive size is False, the fixed number
                                  of classes to divide capacity over.
        """
        if not adaptive_size:
            assert (
                total_num_classes > 0
            ), """When fixed exp mem size, total_num_classes should be > 0."""

        super().__init__(max_size, adaptive_size, total_num_classes)
        self.adaptive_size = adaptive_size
        self.total_num_classes = total_num_classes
        self.seen_classes = set()

    def update(self, strategy: "SupervisedTemplate", **kwargs):
        if isinstance(strategy.experience, OnlineCLExperience):
            new_data = strategy.experience.origin_experience.dataset.eval()
        else:
            new_data = strategy.experience.dataset.eval()

        if len(self.seen_classes.intersection(list(np.unique(new_data.targets))))!= 0:
            # Do not update if we have seen classes already
            return

        new_data_with_logits = dataset_with_logits(
            new_data, strategy.model, strategy.train_mb_size, strategy.device, 100
        )

        # Get sample idxs per class
        cl_idxs = {}
        for idx, target in enumerate(new_data.targets):
            if target not in cl_idxs:
                cl_idxs[target] = []
            cl_idxs[target].append(idx)

        # Make AvalancheSubset per class
        cl_datasets = {}
        for c, c_idxs in cl_idxs.items():
            subset = classification_subset(new_data_with_logits, indices=c_idxs)
            cl_datasets[c] = subset 
        # Update seen classes
        self.seen_classes.update(cl_datasets.keys())

        # associate lengths to classes
        lens = self.get_group_lengths(len(self.seen_classes))
        class_to_len = {}
        for class_id, ll in zip(self.seen_classes, lens):
            class_to_len[class_id] = ll

        # update buffers with new data
        for class_id, new_data_c in cl_datasets.items():
            ll = class_to_len[class_id]
            if class_id in self.buffer_groups:
                old_buffer_c = self.buffer_groups[class_id]
                # Here it uses underlying dataset
                old_buffer_c.update_from_dataset(new_data_c)
                old_buffer_c.resize(strategy, ll)
            else:
                new_buffer = ReservoirSamplingBuffer(ll)
                new_buffer.update_from_dataset(new_data_c)
                self.buffer_groups[class_id] = new_buffer

        # resize buffers
        for class_id, class_buf in self.buffer_groups.items():
            self.buffer_groups[class_id].resize(strategy, class_to_len[class_id])


class DERPlugin(SupervisedPlugin):
    """Also stores previous logits and replays them along with the targets (DER++)"""

    def __init__(
        self,
        mem_size: int = 200,
        batch_size_mem: int = 10,
        adaptive_size=True,
        mode=None,
        alpha=0.1,
        beta=0.5,
    ):
        super().__init__()
        self.batch_size_mem = batch_size_mem
        self.mem_size = mem_size
        self.storage_policy = ClassBalancedBuffer(
            self.mem_size, adaptive_size=adaptive_size
        )
        self.replay_loader = None
        self.mode = mode
        self.alpha = alpha
        self.beta = beta

    def before_training_exp(self, strategy, **kwargs):
        buffer = self.storage_policy.buffer
        if len(buffer) >= self.batch_size_mem:
            self.replay_loader = cycle(
                torch.utils.data.DataLoader(
                    buffer,
                    batch_size=self.batch_size_mem,
                    shuffle=True,
                    drop_last=True,
                )
            )
        else:
            self.replay_loader = None

    def after_training_exp(self, strategy, **kwargs):
        self.storage_policy.update(strategy, **kwargs)

    def before_backward(self, strategy, **kwargs):
        if self.replay_loader is None:
            return None

        batch_x, batch_y, batch_tid, batch_logits, _ = next(self.replay_loader)
        batch_x, batch_y, batch_tid, batch_logits = (
            batch_x.to(strategy.device),
            batch_y.to(strategy.device),
            batch_tid.to(strategy.device),
            batch_logits.to(strategy.device),
        )

        out_buffer = avalanche_forward(strategy.model, batch_x, batch_tid)

        strategy.loss += self.alpha * F.mse_loss(out_buffer, batch_logits[:, :out_buffer.shape[1]])

        if self.mode == "++":
            strategy.loss += self.beta * F.cross_entropy(out_buffer, batch_y)
