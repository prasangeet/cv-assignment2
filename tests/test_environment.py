import importlib


def test_required_dependencies_are_importable():
    """Required libraries should import successfully."""
    dependencies = [
        "torch",
        "numpy",
        "cv2",
        "matplotlib",
    ]

    for dependency in dependencies:
        module = importlib.import_module(dependency)
        assert module is not None, f"Failed to import {dependency}"


def test_pytorch_can_run_on_cpu():
    """Basic tensor operations must work on CPU."""
    import torch

    x = torch.tensor([1.0, 2.0, 3.0], device="cpu")
    y = x * 2

    assert y.device.type == "cpu"
    assert torch.equal(y, torch.tensor([2.0, 4.0, 6.0]))


def test_cpu_only_inference_is_possible():
    """The application must support CPU inference."""
    import torch

    model = torch.nn.Conv2d(3, 1, kernel_size=1).cpu()
    x = torch.randn(1, 3, 16, 16)

    with torch.no_grad():
        output = model(x)

    assert output.device.type == "cpu"
    assert output.shape == (1, 1, 16, 16)
