import numpy as np
import flowiz as fz
from PIL import Image
import torch
from pathlib import Path


def prior_path_for(image_path, prior_root):
    """Mirror <prior-root>/<domain>/<stem>_prior.npy for an *_img1.tif path."""
    p = Path(image_path)
    stem = p.name.rsplit("_img1", 1)[0]
    return Path(prior_root) / p.parent.name / f"{stem}_prior.npy"


class FlowDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_index_list,
        data,
        targets_index_list=None,
        targets=None,
        domain_ids=None,
        sample_uid_by_index=None,
        transform=None,
        train=True,
        prior_root=None,
        prior_channels=0,
    ):
        self.data_index_list = data_index_list
        self.data = data
        self.targets_index_list = targets_index_list
        self.targets = targets
        self.domain_ids = domain_ids
        self.sample_uid_by_index = sample_uid_by_index or {}
        self.transform = transform
        self.count = 0
        self.train = train
        self.prior_root = str(prior_root) if prior_root else None
        self.prior_channels = int(prior_channels)
        self.prior_missing = 0

    def eval(self):
        self.train = False

    def __len__(self):
        return len(self.data_index_list)

    def __getitem__(self, idx):
        sample_ind = self.data_index_list[idx]
        label_ind = self.targets_index_list[idx] if self.targets_index_list is not None else sample_ind

        sample_1 = np.asarray(Image.open(self.data[0][sample_ind])) * 1.0 / 255.0
        sample_2 = np.asarray(Image.open(self.data[1][sample_ind])) * 1.0 / 255.0
        sample = torch.from_numpy(np.array([sample_1, sample_2], dtype=np.float32))

        # The external prior (e.g. a frozen RAFT256-PCK1 flow) is appended as
        # extra input channels: image1, image2, prior[0..C-1].  Callers that do
        # not ask for a prior keep the original 2-channel layout.
        if self.prior_channels > 0 and self.prior_root:
            path = prior_path_for(self.data[0][sample_ind], self.prior_root)
            if path.is_file():
                prior = np.load(path).astype(np.float32)
                if prior.shape[0] > self.prior_channels:
                    prior = prior[: self.prior_channels]
            else:
                self.prior_missing += 1
                prior = np.zeros(
                    (self.prior_channels, sample.shape[-2], sample.shape[-1]), dtype=np.float32
                )
            sample = torch.cat([sample, torch.from_numpy(prior)], dim=0)

        if self.targets is None:
            raise ValueError("FlowDataset requires targets for supervised use.")

        label = fz.read_flow(self.targets[label_ind])
        label_np = label.transpose(2, 0, 1).astype(np.float32)
        label = torch.from_numpy(label_np)

        if self.domain_ids is not None:
            domain_id = torch.tensor(int(self.domain_ids[sample_ind]), dtype=torch.long)
            return sample, label, domain_id
        return sample, label
