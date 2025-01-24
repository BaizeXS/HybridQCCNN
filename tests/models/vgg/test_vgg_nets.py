import pytest
import torch

from components.quanv import AggregationMethod, OutputMode
from models.vgg import HybridVGG, SimpleVGG

# Test parameters
TEST_PARAMS = {
    "batch_size": 2,
    "channels": 3,
    "height": 32,
    "width": 32,
    "num_classes": 10,
}


@pytest.mark.heavy_model
class TestVGG:
    @pytest.fixture
    def sample_input(self):
        """Return a standard test input tensor"""
        return torch.randn(
            TEST_PARAMS["batch_size"],
            TEST_PARAMS["channels"],
            TEST_PARAMS["height"],
            TEST_PARAMS["width"],
        )

    @pytest.fixture
    def simple_vgg(self):
        """Return a default configured SimpleVGG"""
        return SimpleVGG(num_classes=TEST_PARAMS["num_classes"])

    @pytest.fixture
    def hybrid_vgg(self):
        """Return a default configured HybridVGG"""
        return HybridVGG(num_classes=TEST_PARAMS["num_classes"])

    def test_simple_vgg_initialization(self, simple_vgg):
        """Test SimpleVGG initialization parameters"""
        # Test feature extractor
        assert isinstance(simple_vgg.features, torch.nn.Sequential)
        assert isinstance(simple_vgg.avgpool, torch.nn.AdaptiveAvgPool2d)
        assert isinstance(simple_vgg.classifier, torch.nn.Sequential)

        # Test output layer dimension
        assert simple_vgg.classifier[-1].out_features == TEST_PARAMS["num_classes"]

    def test_hybrid_vgg_initialization(self, hybrid_vgg):
        """Test HybridVGG initialization parameters"""
        # Test feature extractor
        assert isinstance(hybrid_vgg.features, torch.nn.Sequential)
        assert isinstance(hybrid_vgg.avgpool, torch.nn.AdaptiveAvgPool2d)
        assert isinstance(hybrid_vgg.classifier, torch.nn.Sequential)

        # Test output layer dimension
        assert hybrid_vgg.classifier[-1].out_features == TEST_PARAMS["num_classes"]

        # Test quantum layer existence
        quantum_layers = [
            layer
            for layer in hybrid_vgg.features
            if isinstance(layer, torch.nn.Module)
            and "Quanv" in layer.__class__.__name__
        ]
        assert len(quantum_layers) > 0

    def test_simple_vgg_forward(self, simple_vgg, sample_input):
        """Test SimpleVGG forward propagation"""
        output = simple_vgg(sample_input)

        # Check output dimension
        expected_shape = (TEST_PARAMS["batch_size"], TEST_PARAMS["num_classes"])
        assert output.shape == expected_shape

        # Check output validity
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_hybrid_vgg_forward(self, hybrid_vgg, sample_input):
        """Test HybridVGG forward propagation"""
        output = hybrid_vgg(sample_input)

        # Check output dimension
        expected_shape = (TEST_PARAMS["batch_size"], TEST_PARAMS["num_classes"])
        assert output.shape == expected_shape

        # Check output validity
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    @pytest.mark.parametrize("output_mode", [OutputMode.QUANTUM, OutputMode.CLASSICAL])
    def test_hybrid_vgg_output_modes(self, sample_input, output_mode):
        """Test HybridVGG different output modes"""
        model = HybridVGG(
            num_classes=TEST_PARAMS["num_classes"], output_mode=output_mode
        )

        output = model(sample_input)
        expected_shape = (TEST_PARAMS["batch_size"], TEST_PARAMS["num_classes"])
        assert output.shape == expected_shape

    @pytest.mark.parametrize(
        "method",
        [AggregationMethod.MEAN, AggregationMethod.SUM, AggregationMethod.WEIGHTED],
    )
    def test_hybrid_vgg_aggregation_methods(self, sample_input, method):
        """Test HybridVGG different aggregation methods"""
        model = HybridVGG(
            num_classes=TEST_PARAMS["num_classes"],
            output_mode=OutputMode.CLASSICAL,
            aggregation_method=method,
        )

        output = model(sample_input)
        expected_shape = (TEST_PARAMS["batch_size"], TEST_PARAMS["num_classes"])
        assert output.shape == expected_shape

    def test_models_output_compatibility(self, simple_vgg, hybrid_vgg, sample_input):
        """Test compatibility of outputs from two models"""
        classic_output = simple_vgg(sample_input)
        hybrid_output = hybrid_vgg(sample_input)

        # Check that the two models output the same shape
        assert classic_output.shape == hybrid_output.shape

    def test_feature_output_shapes(self, simple_vgg, hybrid_vgg, sample_input):
        """Test intermediate feature shapes of both models"""
        # Test feature extractor outputs
        simple_features = simple_vgg.features(sample_input)
        hybrid_features = hybrid_vgg.features(sample_input)
        assert simple_features.shape == hybrid_features.shape

        # Test after avgpool
        simple_pooled = simple_vgg.avgpool(simple_features)
        hybrid_pooled = hybrid_vgg.avgpool(hybrid_features)
        assert simple_pooled.shape == hybrid_pooled.shape
