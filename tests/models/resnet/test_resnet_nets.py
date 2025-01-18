import pytest
import torch

from models.resnet import (
    simple_resnet18, simple_resnet34,
    hybrid_resnet18, hybrid_resnet34
)
from components.quanv import OutputMode, AggregationMethod

# Test parameters
TEST_PARAMS = {
    'batch_size': 2,
    'channels': 3,
    'height': 32,
    'width': 32,
    'num_classes': 10
}

@pytest.fixture
def sample_input():
    """Return a standard test input tensor"""
    return torch.randn(
        TEST_PARAMS['batch_size'],
        TEST_PARAMS['channels'],
        TEST_PARAMS['height'],
        TEST_PARAMS['width']
    )

@pytest.fixture(params=[18, 34])
def simple_resnet(request):
    """Return SimpleResNet models with different depths"""
    if request.param == 18:
        return simple_resnet18(num_classes=TEST_PARAMS['num_classes'])
    return simple_resnet34(num_classes=TEST_PARAMS['num_classes'])

@pytest.fixture(params=[18, 34])
def hybrid_resnet(request):
    """Return HybridResNet models with different depths"""
    if request.param == 18:
        return hybrid_resnet18(num_classes=TEST_PARAMS['num_classes'])
    return hybrid_resnet34(num_classes=TEST_PARAMS['num_classes'])

def test_simple_resnet_initialization(simple_resnet):
    """Test SimpleResNet initialization parameters"""
    # Test basic components
    assert isinstance(simple_resnet.conv1, torch.nn.Conv2d)
    assert isinstance(simple_resnet.bn1, torch.nn.BatchNorm2d)
    assert isinstance(simple_resnet.layer1, torch.nn.Sequential)
    assert isinstance(simple_resnet.fc, torch.nn.Linear)
    
    # Test output layer dimension
    assert simple_resnet.fc.out_features == TEST_PARAMS['num_classes']

def test_hybrid_resnet_initialization(hybrid_resnet):
    """Test HybridResNet initialization parameters"""
    # Test basic components
    assert isinstance(hybrid_resnet.conv1, torch.nn.Conv2d)
    assert isinstance(hybrid_resnet.bn1, torch.nn.BatchNorm2d)
    assert isinstance(hybrid_resnet.layer1, torch.nn.Sequential)
    assert isinstance(hybrid_resnet.fc, torch.nn.Linear)
    
    # Test output layer dimension
    assert hybrid_resnet.fc.out_features == TEST_PARAMS['num_classes']
    
    # Test quantum layer existence
    quantum_layers = []
    for module in hybrid_resnet.modules():
        if hasattr(module, 'quanv'):
            quantum_layers.append(module)
    assert len(quantum_layers) > 0

def test_simple_resnet_forward(simple_resnet, sample_input):
    """Test SimpleResNet forward propagation"""
    output = simple_resnet(sample_input)
    
    # Check output dimension
    expected_shape = (TEST_PARAMS['batch_size'], TEST_PARAMS['num_classes'])
    assert output.shape == expected_shape
    
    # Check output validity
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()

def test_hybrid_resnet_forward(hybrid_resnet, sample_input):
    """Test HybridResNet forward propagation"""
    output = hybrid_resnet(sample_input)
    
    # Check output dimension
    expected_shape = (TEST_PARAMS['batch_size'], TEST_PARAMS['num_classes'])
    assert output.shape == expected_shape
    
    # Check output validity
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()

@pytest.mark.parametrize("output_mode", [
    OutputMode.QUANTUM,
    OutputMode.CLASSICAL
])
def test_hybrid_resnet_output_modes(sample_input, output_mode):
    """Test HybridResNet different output modes"""
    model = hybrid_resnet18(
        num_classes=TEST_PARAMS['num_classes'],
        output_mode=output_mode
    )
    
    output = model(sample_input)
    expected_shape = (TEST_PARAMS['batch_size'], TEST_PARAMS['num_classes'])
    assert output.shape == expected_shape

@pytest.mark.parametrize("method", [
    AggregationMethod.MEAN,
    AggregationMethod.SUM,
    AggregationMethod.WEIGHTED
])
def test_hybrid_resnet_aggregation_methods(sample_input, method):
    """Test HybridResNet different aggregation methods"""
    model = hybrid_resnet18(
        num_classes=TEST_PARAMS['num_classes'],
        output_mode=OutputMode.CLASSICAL,
        aggregation_method=method
    )
    
    output = model(sample_input)
    expected_shape = (TEST_PARAMS['batch_size'], TEST_PARAMS['num_classes'])
    assert output.shape == expected_shape

def test_models_output_compatibility(simple_resnet, hybrid_resnet, sample_input):
    """Test compatibility of outputs from two models"""
    classic_output = simple_resnet(sample_input)
    hybrid_output = hybrid_resnet(sample_input)
    
    # Check that the two models output the same shape
    assert classic_output.shape == hybrid_output.shape

def test_residual_connections(simple_resnet, hybrid_resnet, sample_input):
    """Test residual connections in both models"""
    # Get intermediate features before and after residual connections
    def get_residual_output(model, x):
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        return model.layer1(x)  # First residual block
    
    simple_res = get_residual_output(simple_resnet, sample_input)
    hybrid_res = get_residual_output(hybrid_resnet, sample_input)
    
    # Check shapes match
    assert simple_res.shape == hybrid_res.shape
    
    # Check outputs are valid
    assert not torch.isnan(simple_res).any()
    assert not torch.isnan(hybrid_res).any() 