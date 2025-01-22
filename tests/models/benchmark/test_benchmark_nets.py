import pytest
import torch

from components.quanv import AggregationMethod, OutputMode
from models.benchmark import ClassicNet, HybridNet

# Test parameters
TEST_PARAMS = {
    "batch_size": 2,
    "channels": 1,
    "height": 28,
    "width": 28,
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
def classic_net():
    """Return a default configured ClassicNet"""
    return ClassicNet(num_classes=TEST_PARAMS["num_classes"])


@pytest.fixture
def hybrid_net():
    """Return a default configured HybridNet"""
    return HybridNet(num_classes=TEST_PARAMS["num_classes"])


def test_classic_net_initialization(classic_net):
    """Test ClassicNet initialization parameters"""
    assert isinstance(classic_net.conv1, torch.nn.Conv2d)
    assert isinstance(classic_net.conv2, torch.nn.Conv2d)
    assert isinstance(classic_net.fc1, torch.nn.Linear)
    assert isinstance(classic_net.fc2, torch.nn.Linear)

    # Test output layer dimension
    assert classic_net.fc2.out_features == TEST_PARAMS["num_classes"]


def test_hybrid_net_initialization(hybrid_net):
    """Test HybridNet initialization parameters"""
    assert isinstance(hybrid_net.quanv, torch.nn.Module)
    assert isinstance(hybrid_net.conv, torch.nn.Conv2d)
    assert isinstance(hybrid_net.fc1, torch.nn.Linear)
    assert isinstance(hybrid_net.fc2, torch.nn.Linear)

    # Test output layer dimension
    assert hybrid_net.fc2.out_features == TEST_PARAMS["num_classes"]


def test_classic_net_forward(classic_net, sample_input):
    """Test ClassicNet forward propagation"""
    output = classic_net(sample_input)

    # Check output dimension
    expected_shape = (TEST_PARAMS["batch_size"], TEST_PARAMS["num_classes"])
    assert output.shape == expected_shape

    # Check output validity
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_hybrid_net_forward(hybrid_net, sample_input):
    """Test HybridNet forward propagation"""
    output = hybrid_net(sample_input)

    # Check output dimension
    expected_shape = (TEST_PARAMS["batch_size"], TEST_PARAMS["num_classes"])
    assert output.shape == expected_shape

    # Check output validity
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


@pytest.mark.parametrize("output_mode", [OutputMode.QUANTUM, OutputMode.CLASSICAL])
def test_hybrid_net_output_modes(sample_input, output_mode):
    """Test HybridNet different output modes"""
    model = HybridNet(num_classes=TEST_PARAMS["num_classes"], output_mode=output_mode)

    output = model(sample_input)
    expected_shape = (TEST_PARAMS["batch_size"], TEST_PARAMS["num_classes"])
    assert output.shape == expected_shape


@pytest.mark.parametrize(
    "method",
    [AggregationMethod.MEAN, AggregationMethod.SUM, AggregationMethod.WEIGHTED],
)
def test_hybrid_net_aggregation_methods(sample_input, method):
    """Test HybridNet different aggregation methods"""
    model = HybridNet(
        num_classes=TEST_PARAMS["num_classes"],
        output_mode=OutputMode.CLASSICAL,
        aggregation_method=method,
    )

    output = model(sample_input)
    expected_shape = (TEST_PARAMS["batch_size"], TEST_PARAMS["num_classes"])
    assert output.shape == expected_shape


def test_models_output_compatibility(classic_net, hybrid_net, sample_input):
    """Test compatibility of outputs from two models"""
    classic_output = classic_net(sample_input)
    hybrid_output = hybrid_net(sample_input)

    # Check that the two models output the same shape
    assert classic_output.shape == hybrid_output.shape
