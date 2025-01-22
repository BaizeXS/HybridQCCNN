import pytest
import torch

from components.quanv import AggregationMethod, OutputMode
from models.googlenet import HybridGoogLeNet, SimpleGoogLeNet

# Test parameters
TEST_PARAMS = {
    "batch_size": 2,
    "channels": 3,
    "height": 32,
    "width": 32,
    "num_classes": 10,
}


@pytest.fixture
def sample_input():
    """Return a standard test input tensor"""
    return torch.randn(
        TEST_PARAMS["batch_size"],
        TEST_PARAMS["channels"],
        TEST_PARAMS["height"],
        TEST_PARAMS["width"],
    )


@pytest.fixture
def simple_googlenet():
    """Return a default configured SimpleGoogLeNet"""
    return SimpleGoogLeNet(num_classes=TEST_PARAMS["num_classes"])


@pytest.fixture
def hybrid_googlenet():
    """Return a default configured HybridGoogLeNet"""
    return HybridGoogLeNet(num_classes=TEST_PARAMS["num_classes"])


def test_simple_googlenet_initialization(simple_googlenet):
    """Test SimpleGoogLeNet initialization parameters"""
    # Test basic components
    assert isinstance(simple_googlenet.conv1, torch.nn.Module)
    assert isinstance(simple_googlenet.inception3a, torch.nn.Module)
    assert isinstance(simple_googlenet.inception4b, torch.nn.Module)
    assert isinstance(simple_googlenet.fc, torch.nn.Linear)

    # Test output layer dimension
    assert simple_googlenet.fc.out_features == TEST_PARAMS["num_classes"]


def test_hybrid_googlenet_initialization(hybrid_googlenet):
    """Test HybridGoogLeNet initialization parameters"""
    # Test basic components
    assert isinstance(hybrid_googlenet.conv1, torch.nn.Module)
    assert isinstance(hybrid_googlenet.inception3a, torch.nn.Module)
    assert isinstance(hybrid_googlenet.inception4b, torch.nn.Module)
    assert isinstance(hybrid_googlenet.fc, torch.nn.Linear)

    # Test output layer dimension
    assert hybrid_googlenet.fc.out_features == TEST_PARAMS["num_classes"]

    # Test quantum layer existence
    quantum_layers = []
    for module in hybrid_googlenet.modules():
        if hasattr(module, "quanv"):
            quantum_layers.append(module)
    assert len(quantum_layers) > 0


def test_simple_googlenet_forward(simple_googlenet, sample_input):
    """Test SimpleGoogLeNet forward propagation"""
    output = simple_googlenet(sample_input)

    # Handle auxiliary outputs during training
    if isinstance(output, tuple):
        output, aux = output
        # Check auxiliary output dimension
        assert aux.shape == (TEST_PARAMS["batch_size"], TEST_PARAMS["num_classes"])

    # Check output dimension
    expected_shape = (TEST_PARAMS["batch_size"], TEST_PARAMS["num_classes"])
    assert output.shape == expected_shape

    # Check output validity
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_hybrid_googlenet_forward(hybrid_googlenet, sample_input):
    """Test HybridGoogLeNet forward propagation"""
    output = hybrid_googlenet(sample_input)

    # Handle auxiliary outputs during training
    if isinstance(output, tuple):
        output, aux = output
        # Check auxiliary output dimension
        assert aux.shape == (TEST_PARAMS["batch_size"], TEST_PARAMS["num_classes"])

    # Check output dimension
    expected_shape = (TEST_PARAMS["batch_size"], TEST_PARAMS["num_classes"])
    assert output.shape == expected_shape

    # Check output validity
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


@pytest.mark.parametrize("output_mode", [OutputMode.QUANTUM, OutputMode.CLASSICAL])
def test_hybrid_googlenet_output_modes(sample_input, output_mode):
    """Test HybridGoogLeNet different output modes"""
    model = HybridGoogLeNet(
        num_classes=TEST_PARAMS["num_classes"],
        aux_logits=False,
        output_mode=output_mode,
    )

    output = model(sample_input)
    expected_shape = (TEST_PARAMS["batch_size"], TEST_PARAMS["num_classes"])
    assert output.shape == expected_shape


@pytest.mark.parametrize(
    "method",
    [AggregationMethod.MEAN, AggregationMethod.SUM, AggregationMethod.WEIGHTED],
)
def test_hybrid_googlenet_aggregation_methods(sample_input, method):
    """Test HybridGoogLeNet different aggregation methods"""
    model = HybridGoogLeNet(
        num_classes=TEST_PARAMS["num_classes"],
        aux_logits=False,
        output_mode=OutputMode.CLASSICAL,
        aggregation_method=method,
    )

    output = model(sample_input)
    expected_shape = (TEST_PARAMS["batch_size"], TEST_PARAMS["num_classes"])
    assert output.shape == expected_shape


def test_models_output_compatibility(simple_googlenet, hybrid_googlenet, sample_input):
    """Test compatibility of outputs from two models"""
    # Set both models to eval mode to avoid auxiliary outputs
    simple_googlenet.eval()
    hybrid_googlenet.eval()

    classic_output = simple_googlenet(sample_input)
    hybrid_output = hybrid_googlenet(sample_input)

    # Check that the two models output the same shape
    assert classic_output.shape == hybrid_output.shape


def test_auxiliary_outputs():
    """Test auxiliary classifier outputs during training"""
    model = SimpleGoogLeNet(num_classes=TEST_PARAMS["num_classes"], aux_logits=True)
    model.train()  # Set to training mode

    x = torch.randn(
        TEST_PARAMS["batch_size"],
        TEST_PARAMS["channels"],
        TEST_PARAMS["height"],
        TEST_PARAMS["width"],
    )

    output, aux = model(x)
    assert output.shape == (TEST_PARAMS["batch_size"], TEST_PARAMS["num_classes"])
    assert aux.shape == (TEST_PARAMS["batch_size"], TEST_PARAMS["num_classes"])
