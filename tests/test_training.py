import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.train import run_epoch


class TinySegmentationModel(nn.Module):
    """Small model for fast training tests."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


def make_loader():
    images = torch.rand(4, 3, 16, 16)
    masks = torch.randint(0, 2, (4, 1, 16, 16)).float()

    return DataLoader(
        TensorDataset(images, masks),
        batch_size=2,
    )


def test_training_epoch_updates_parameters():
    torch.manual_seed(42)

    model = TinySegmentationModel()
    loader = make_loader()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    before = model.conv.weight.detach().clone()

    loss = run_epoch(
        model,
        loader,
        criterion,
        torch.device("cpu"),
        optimizer,
    )

    after = model.conv.weight.detach()

    assert loss > 0
    assert torch.isfinite(torch.tensor(loss))
    assert not torch.equal(before, after)


def test_validation_epoch_does_not_update_parameters():
    model = TinySegmentationModel()
    loader = make_loader()
    criterion = nn.BCEWithLogitsLoss()

    before = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }

    loss = run_epoch(
        model,
        loader,
        criterion,
        torch.device("cpu"),
    )

    assert loss > 0
    assert torch.isfinite(torch.tensor(loss))

    for name, value in model.state_dict().items():
        assert torch.equal(before[name], value)


def test_training_epoch_returns_finite_loss():
    model = TinySegmentationModel()
    loader = make_loader()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters())

    loss = run_epoch(
        model,
        loader,
        criterion,
        torch.device("cpu"),
        optimizer,
    )

    assert torch.isfinite(torch.tensor(loss))
