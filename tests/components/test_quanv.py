import pytest
import torch

from components import Quanv2d, OutputMode, AggregationMethod

# Test parameters
TEST_PARAMS = {
    'batch_size': 2,
    'in_channels': 2,
    'out_channels': 4,
    'kernel_size': 2,
    'stride': 1,
    'padding': 1,
    'height': 14,
    'width': 14,
    'small_size': 8
}


@pytest.fixture
def default_layer():
    """Return a Quanv2d layer with default configuration"""
    return Quanv2d(
        in_channels=TEST_PARAMS['in_channels'],
        out_channels=TEST_PARAMS['out_channels'],
        kernel_size=TEST_PARAMS['kernel_size'],
        stride=TEST_PARAMS['stride'],
        padding=TEST_PARAMS['padding']
    )


@pytest.fixture
def sample_input():
    """Return a standard test input tensor"""
    return torch.randn(
        TEST_PARAMS['batch_size'],
        TEST_PARAMS['in_channels'],
        TEST_PARAMS['height'],
        TEST_PARAMS['width']
    )


def test_quanv2d_initialization(default_layer):
    """Test Quanv2d initialization parameters"""
    # Test basic parameters
    assert default_layer.in_channels == TEST_PARAMS['in_channels']
    assert default_layer.out_channels == TEST_PARAMS['out_channels']
    assert default_layer.kernel_size == (TEST_PARAMS['kernel_size'], TEST_PARAMS['kernel_size'])
    assert default_layer.stride == (TEST_PARAMS['stride'], TEST_PARAMS['stride'])
    assert default_layer.padding == (TEST_PARAMS['padding'], TEST_PARAMS['padding'])

    # Test default values
    assert default_layer.device == "cpu"
    assert default_layer.output_mode == OutputMode.QUANTUM
    assert default_layer.aggregation_method == AggregationMethod.MEAN
    assert default_layer.preserve_quantum_info is False


@pytest.mark.parametrize("invalid_param", [
    {'kernel_size': 0, 'error_match': "kernel_size must be positive"},
    {'in_channels': 0, 'error_match': "in_channels must be positive"},
    {'out_channels': 0, 'error_match': "out_channels must be positive"},
    {'stride': 0, 'error_match': "stride must be positive"},
    {'padding': -1, 'error_match': "padding cannot be negative"},
])
def test_invalid_parameters(invalid_param):
    """Test invalid parameters"""
    params = {
        'in_channels': TEST_PARAMS['in_channels'],
        'out_channels': TEST_PARAMS['out_channels'],
        'kernel_size': TEST_PARAMS['kernel_size'],
        'stride': TEST_PARAMS['stride'],
        'padding': TEST_PARAMS['padding']
    }

    # Update invalid parameters
    for key, value in invalid_param.items():
        if key != 'error_match':
            params[key] = value

    with pytest.raises(ValueError, match=invalid_param['error_match']):
        Quanv2d(**params)


def test_quanv2d_input_validation(default_layer):
    """Test input tensor dimension validation"""
    # Valid input
    valid_input = torch.randn(
        TEST_PARAMS['batch_size'],
        TEST_PARAMS['in_channels'],
        TEST_PARAMS['height'],
        TEST_PARAMS['width']
    )

    h_out = calculate_output_size(TEST_PARAMS['height'])
    w_out = calculate_output_size(TEST_PARAMS['width'])

    output = default_layer(valid_input)
    assert output.shape == (TEST_PARAMS['batch_size'], TEST_PARAMS['out_channels'], h_out, w_out)

    # Invalid dimension input
    with pytest.raises(ValueError, match="Expected 4D input tensor"):
        default_layer(torch.randn(1, 2, 14))

    # Invalid channel number
    with pytest.raises(ValueError, match=f"Expected {TEST_PARAMS['in_channels']} channels"):
        default_layer(torch.randn(1, 3, 14, 14))


@pytest.mark.parametrize("output_mode,preserve_info", [
    (OutputMode.QUANTUM, True),
    (OutputMode.QUANTUM, False),
    (OutputMode.CLASSICAL, False)
])
def test_output_modes(output_mode, preserve_info):
    """Test different output modes"""
    x = torch.randn(1, TEST_PARAMS['in_channels'], TEST_PARAMS['small_size'], TEST_PARAMS['small_size'])

    h_out = calculate_output_size(TEST_PARAMS['small_size'])
    w_out = calculate_output_size(TEST_PARAMS['small_size'])

    layer = Quanv2d(
        in_channels=TEST_PARAMS['in_channels'],
        out_channels=TEST_PARAMS['out_channels'],
        kernel_size=TEST_PARAMS['kernel_size'],
        stride=TEST_PARAMS['stride'],
        padding=TEST_PARAMS['padding'],
        output_mode=output_mode,
        preserve_quantum_info=preserve_info
    )

    output = layer(x)
    expected_shape = (1, TEST_PARAMS['out_channels'], h_out, w_out)
    assert output.shape == expected_shape


@pytest.mark.parametrize("method", [
    AggregationMethod.MEAN,
    AggregationMethod.SUM,
    AggregationMethod.WEIGHTED
])
def test_aggregation_methods(method):
    """Test different aggregation methods in CLASSICAL mode"""
    x = torch.randn(1, TEST_PARAMS['in_channels'], TEST_PARAMS['small_size'], TEST_PARAMS['small_size'])

    h_out = calculate_output_size(TEST_PARAMS['small_size'])
    w_out = calculate_output_size(TEST_PARAMS['small_size'])

    # Basic configuration
    layer_config = {
        'in_channels': TEST_PARAMS['in_channels'],
        'out_channels': TEST_PARAMS['out_channels'],
        'kernel_size': TEST_PARAMS['kernel_size'],
        'stride': TEST_PARAMS['stride'],
        'padding': TEST_PARAMS['padding'],
        'output_mode': OutputMode.CLASSICAL,  # Ensure classical mode
        'aggregation_method': method
    }

    # For WEIGHTED method, add extra configuration
    if method == AggregationMethod.WEIGHTED:
        layer_config.update({
            'preserve_quantum_info': False,  # Ensure weights are initialized correctly
            'qdevice': 'default.qubit'  # Add quantum device configuration
        })

    layer = Quanv2d(**layer_config)

    output = layer(x)
    expected_shape = (1, TEST_PARAMS['out_channels'], h_out, w_out)
    assert output.shape == expected_shape
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def calculate_output_size(input_size):
    """Calculate output size for convolution"""
    padding = TEST_PARAMS['padding']
    kernel_size = TEST_PARAMS['kernel_size']
    stride = TEST_PARAMS['stride']
    return (input_size + 2 * padding - kernel_size) // stride + 1
