import torch
from torch import nn

from models.segmentation_cnn import SegmentationCNN


def test_model_initializes():
    model = SegmentationCNN()
    assert isinstance(model, nn.Module)


def test_model_output_shape(model, sample_image):
    """One RGB image should produce one full-resolution mask."""
    model.eval()

    with torch.no_grad():
        output = model(sample_image)

    assert output.shape == (1, 1, 256, 256)


def test_model_supports_batch_input(model, small_image_batch):
    """The model should process multiple images in one forward pass."""
    model.eval()

    with torch.no_grad():
        output = model(small_image_batch)

    assert output.shape == (2, 1, 256, 256)


def test_model_output_is_finite(model, sample_image):
    """The forward pass must not produce NaN or infinity."""
    model.eval()

    with torch.no_grad():
        output = model(sample_image)

    assert torch.isfinite(output).all()


def test_model_has_trainable_parameters(model):
    """The network must contain trainable parameters."""
    parameters = list(model.parameters())

    assert parameters
    assert all(parameter.requires_grad for parameter in parameters)

    total_parameters = sum(
        parameter.numel() for parameter in parameters
    )

    assert 0 < total_parameters < 5_000_000


def test_model_supports_backward_pass(sample_image):
    """Gradients should flow through the network."""
    model = SegmentationCNN()
    model.train()

    target = torch.randint(
        0, 2, (1, 1, 256, 256)
    ).float()

    output = model(sample_image)

    loss = nn.BCEWithLogitsLoss()(output, target)
    loss.backward()

    assert torch.isfinite(loss)
    assert loss.item() > 0

    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    assert all(gradient is not None for gradient in gradients)
    assert any(torch.count_nonzero(g).item() > 0 for g in gradients)


def test_model_can_switch_between_train_and_eval(model):
    model.train()
    assert model.training

    model.eval()
    assert not model.training
