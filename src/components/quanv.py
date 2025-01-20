"""Quantum convolution layer implementation.

This module provides:
1. Quantum convolution layer (Quanv2d)
2. Output modes for quantum measurements
3. Methods for aggregating quantum outputs
"""

from enum import Enum
from typing import Optional, Dict, Any, Tuple, Union

import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F

from .qkernel import QKernel

__all__ = ["_QuanvNd", "Quanv2d", "OutputMode", "AggregationMethod"]


class OutputMode(str, Enum):
    """Quantum convolution output modes.
    
    Modes:
        QUANTUM: Keep full quantum output
        CLASSICAL: Aggregate quantum outputs
    """
    QUANTUM = 'quantum'
    CLASSICAL = 'classical'


class AggregationMethod(str, Enum):
    """Methods for aggregating quantum outputs.
    
    Methods:
        MEAN: Average of quantum measurements
        SUM: Sum of quantum measurements
        WEIGHTED: Weighted sum of quantum measurements
    """
    MEAN = 'mean'
    SUM = 'sum'
    WEIGHTED = 'weighted'


class _QuanvNd(nn.Module):
    """Base class for N-dimensional quantum convolution layers.
    
    This class provides:
    - Basic parameter validation
    - Input preprocessing
    - Quantum output processing
    - Output aggregation methods
    
    Attributes:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (Tuple[int, ...]): Size of convolution kernel.
        stride (Tuple[int, ...]): Convolution stride.
        padding (Tuple[int, ...]): Convolution padding.
        device (str): Computing device.
        output_mode (OutputMode): Quantum output mode.
        aggregation_method (AggregationMethod): Output aggregation method.
    """

    EPSILON = 1e-8  # Used to avoid division by zero

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, ...]],
                 stride: Union[int, Tuple[int, ...]],
                 padding: Union[int, Tuple[int, ...]],
                 device: str = "cpu",
                 qkernel: Optional[QKernel] = None,
                 num_qlayers: int = 2,
                 qdevice: str = "default.qubit",
                 qdevice_kwargs: Optional[Dict[str, Any]] = None,
                 diff_method: str = "best",
                 output_mode: Union[OutputMode, str] = OutputMode.QUANTUM,
                 aggregation_method: Union[AggregationMethod, str] = AggregationMethod.MEAN,
                 preserve_quantum_info: bool = False
                 ):
        """Initialize quantum convolution layer.
        
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (Union[int, Tuple[int, ...]]): Size of convolution kernel.
            stride (Union[int, Tuple[int, ...]]): Convolution stride.
            padding (Union[int, Tuple[int, ...]]): Convolution padding.
            device (str): Computing device.
            qkernel (Optional[QKernel]): Custom quantum kernel.
            num_qlayers (int): Number of quantum layers.
            qdevice (str): Quantum device name.
            qdevice_kwargs (Optional[Dict[str, Any]]): Quantum device parameters.
            diff_method (str): Differentiation method.
            output_mode (Union[OutputMode, str]): Quantum output mode.
            aggregation_method (Union[AggregationMethod, str]): Output aggregation method.
            preserve_quantum_info (bool): Whether to preserve quantum information.
        """
        super().__init__()

        # Validate basic parameters first
        self._validate_basic_params(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

        # Classical convolution parameters  
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = self._to_tuple(kernel_size, n=2)
        self.stride = self._to_tuple(stride, n=2)
        self.padding = self._to_tuple(padding, n=2)
        self.device = device

        # Output mode and aggregation method configuration
        self.output_mode = OutputMode(output_mode)
        self.aggregation_method = AggregationMethod(aggregation_method)

        # Preserve quantum info setting
        self.preserve_quantum_info = preserve_quantum_info

        # Initialize quantum components  
        self._setup_quantum_components(qkernel, num_qlayers, qdevice, qdevice_kwargs, diff_method)

        # Initialize aggregation weights if needed  
        self._setup_aggregation_weights()

        # Used to store the last input  
        self.last_input = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Abstract forward method - to be implemented by subclasses"""
        raise NotImplementedError("Forward method must be implemented in subclasses")

    def _validate_basic_params(self, **kwargs):
        """Validate basic convolution parameters.
        
        Args:
            **kwargs: Dictionary containing parameters to validate:
                - in_channels (int): Number of input channels
                - out_channels (int): Number of output channels
                - kernel_size (Union[int, Tuple]): Convolution kernel size
                - stride (Union[int, Tuple]): Convolution stride
                - padding (Union[int, Tuple]): Convolution padding
                
        Raises:
            ValueError: If any parameter is invalid
        """
        # Validate in_channels
        if not isinstance(kwargs['in_channels'], int) or kwargs['in_channels'] <= 0:
            raise ValueError("in_channels must be positive")

        # Validate out_channels
        if not isinstance(kwargs['out_channels'], int) or kwargs['out_channels'] <= 0:
            raise ValueError("out_channels must be positive")

        # Validate kernel_size
        if isinstance(kwargs['kernel_size'], int):
            if kwargs['kernel_size'] <= 0:
                raise ValueError("kernel_size must be positive")
        elif isinstance(kwargs['kernel_size'], tuple):
            if any(not isinstance(k, int) or k <= 0 for k in kwargs['kernel_size']):
                raise ValueError("kernel_size must be positive")
        else:
            raise ValueError("kernel_size must be an integer or tuple")

        # Validate stride
        if isinstance(kwargs['stride'], int):
            if kwargs['stride'] <= 0:
                raise ValueError("stride must be positive")
        elif isinstance(kwargs['stride'], tuple):
            if any(not isinstance(s, int) or s <= 0 for s in kwargs['stride']):
                raise ValueError("stride must be positive")
        else:
            raise ValueError("stride must be an integer or tuple")

        # Validate padding
        if isinstance(kwargs['padding'], int):
            if kwargs['padding'] < 0:
                raise ValueError("padding cannot be negative")
        elif isinstance(kwargs['padding'], tuple):
            if any(not isinstance(p, int) or p < 0 for p in kwargs['padding']):
                raise ValueError("padding cannot be negative")
        else:
            raise ValueError("padding must be an integer or tuple")

    def _validate_input(self, x: torch.Tensor) -> None:
        """Validate input tensor; this will be implemented in subclasses"""
        raise NotImplementedError("Input validation must be implemented in subclasses")

    def _setup_quantum_components(self, qkernel, num_qlayers, qdevice, qdevice_kwargs, diff_method):
        """Initialize quantum components of the layer.
        
        Args:
            qkernel (Optional[QKernel]): Custom quantum kernel
            num_qlayers (int): Number of quantum layers
            qdevice (str): Quantum device name
            qdevice_kwargs (Optional[Dict]): Quantum device parameters
            diff_method (str): Differentiation method
        """
        if qkernel is not None:
            # Use Custom QKernel
            expected_qubits = self.in_channels * torch.prod(torch.tensor(self.kernel_size)).item()
            if qkernel.num_qubits != expected_qubits:
                raise ValueError(f"Quantum kernel must have {expected_qubits} qubits, but got {qkernel.num_qubits}")
            self.qkernel = qkernel
        else:
            # Use Default QKernel
            self.qkernel = QKernel(
                quantum_channels=self.in_channels,
                kernel_size=self.kernel_size,
                num_param_blocks=num_qlayers
            )

            # Quantum device setup
        self.qdevice_kwargs = qdevice_kwargs or {}
        self.qdevice = qml.device(qdevice, wires=self.qkernel.num_qubits, **self.qdevice_kwargs)

        # Quantum neural network setup  
        self.qnode = qml.QNode(self.qkernel.circuit, device=self.qdevice, interface="torch", diff_method=diff_method)
        self.qlayer = qml.qnn.TorchLayer(self.qnode, self.qkernel.weight_shapes)

    def _setup_aggregation_weights(self) -> None:
        """Initialize weights for weighted aggregation method.
        
        Creates trainable weights if using WEIGHTED aggregation method.
        """
        if self.aggregation_method == AggregationMethod.WEIGHTED:
            weight_shape = (
                self.in_channels,  # Input channels
                torch.prod(torch.tensor(self.kernel_size)).item()  # Weight vector size
            )
            self.aggregation_weights = nn.Parameter(torch.empty(weight_shape))
            nn.init.xavier_uniform_(self.aggregation_weights)

    def _to_tuple(self, x: Union[int, Tuple], n: int = 2) -> Tuple:
        """Convert input to n-dimensional tuple.
        
        Args:
            x (Union[int, Tuple]): Input value to convert
            n (int): Target tuple dimension
            
        Returns:
            Tuple: n-dimensional tuple
        """
        if isinstance(x, tuple) and len(x) != n:
            raise ValueError(f"Expected tuple of length {n}, got length {len(x)}")
        return (x,) * n if isinstance(x, int) else x


class Quanv2d(_QuanvNd):
    """2D quantum convolution layer.
    
    This class implements a 2D quantum convolution operation that:
    - Processes input data through a quantum circuit
    - Applies quantum measurements
    - Aggregates quantum outputs into classical format
    
    The layer supports different output modes and aggregation methods.
    
    Attributes:
        All attributes inherited from _QuanvNd base class
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Set input channels based on preserve_quantum_info and output_mode
        conv_in_channels = (
            self.in_channels * self.kernel_size[0] * self.kernel_size[1]
            if self.preserve_quantum_info and self.output_mode == OutputMode.QUANTUM
            else self.in_channels
        )

        # Initialize 1x1 convolution layer
        self.classical_conv = nn.Conv2d(
            in_channels=conv_in_channels,
            out_channels=self.out_channels,
            kernel_size=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the quantum convolution layer.
        
        Process steps:
        1. Input validation and preprocessing
        2. Quantum circuit computation
        3. Output processing based on mode
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
            
        Returns:
            torch.Tensor: Processed output tensor
            
        Raises:
            ValueError: If input dimensions are invalid
        """
        # Validate input
        self._validate_input(x)
        return self._quantum_conv(x)

    def _validate_input(self, x: torch.Tensor) -> None:
        """Check if input dimensions are as expected for 2D"""
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input tensor for 2D convolution, got {x.dim()}D")
        if x.size(1) != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {x.size(1)}")

    def _quantum_conv(self, x: torch.Tensor) -> torch.Tensor:
        """Implement the 2D quantum convolution operation"""
        # 1. Preprocess input and calculate output dimensions
        x = self._preprocess_input(x)
        bs, c, h, w = x.shape
        h_out = (h + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        w_out = (w + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1

        # 2. Extract patches
        patches = F.unfold(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding
        ).permute(0, 2, 1)  # (B, L, C*K*K)

        # 3. Apply quantum layer to each patch
        quantum_outputs = []
        for patch in patches.reshape(-1, patches.shape[-1]):
            quantum_output = self.qlayer(patch)
            quantum_outputs.append(quantum_output)
        quantum_outputs = torch.stack(quantum_outputs).reshape(bs, patches.shape[1], -1)  # (B, L, C*K*K)

        # 4. Process based on mode
        if self.output_mode == OutputMode.QUANTUM:
            if self.preserve_quantum_info:
                # Retain all quantum outputs: (B, C*K*K, h_out, w_out)
                out = quantum_outputs.permute(0, 2, 1).reshape(bs, -1, h_out, w_out)
            else:
                # Do not retain all quantum outputs: (B, C, h_out, w_out)
                out = self._process_quantum_incomplete(quantum_outputs, x, h_out, w_out)
        else:
            # Classical Mode using quantum outputs for aggregation: (B, C, h_out, w_out)
            out = self._process_classical_mode(quantum_outputs, bs, c, h_out, w_out)

        # 5. Channel adjustment
        out = self.classical_conv(out)
        return out

    def _preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input data to [-1, 1] range for each batch.
        
        Args:
            x (torch.Tensor): Input tensor to normalize
            
        Returns:
            torch.Tensor: Normalized input tensor in range [-1, 1]
        """
        x_min = x.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
        x_max = x.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
        # Linear mapping
        x_normalized = (2 * (x - x_min) / (x_max - x_min + self.EPSILON)) - 1
        return x_normalized

    def _process_quantum_incomplete(self, quantum_outputs: torch.Tensor, x: torch.Tensor, h_out: int,
                                    w_out: int) -> torch.Tensor:
        """Process incomplete quantum outputs.
        
        Args:
            quantum_outputs (torch.Tensor): Raw quantum measurements
            x (torch.Tensor): Original input tensor
            h_out (int): Target output height
            w_out (int): Target output width
            
        Returns:
            torch.Tensor: Processed quantum output tensor
        """
        # Gain original input shape
        _, _, h_in, w_in = x.shape

        # Fold to original shape
        out = F.fold(
            quantum_outputs.permute(0, 2, 1),
            output_size=(h_in, w_in),
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding
        )

        # Apply normalization
        norm_factor = self._compute_normalization_factor(x)
        out = out / (norm_factor + self.EPSILON)

        # If needed, use adaptive average pooling to adjust output size
        if (h_in, w_in) != (h_out, w_out):
            out = F.adaptive_avg_pool2d(out, (h_out, w_out))

        return out

    def _process_classical_mode(self, quantum_outputs: torch.Tensor, bs: int, c: int, h_out: int,
                                w_out: int) -> torch.Tensor:
        """Process quantum outputs in classical mode.
        
        Args:
            quantum_outputs (torch.Tensor): Raw quantum measurements
            bs (int): Batch size
            c (int): Number of channels
            h_out (int): Output height
            w_out (int): Output width
            
        Returns:
            torch.Tensor: Processed classical output
        """
        # Reshape to (B, num_patches, C, K*K)
        quantum_outputs = quantum_outputs.view(bs, -1, c, self.kernel_size[0] * self.kernel_size[1])

        # Aggregate based on selected method
        if self.aggregation_method == AggregationMethod.MEAN:
            out = quantum_outputs.mean(dim=-1)  # (B, num_patches, C)
        elif self.aggregation_method == AggregationMethod.SUM:
            out = quantum_outputs.sum(dim=-1)  # (B, num_patches, C)
        elif self.aggregation_method == AggregationMethod.WEIGHTED:
            # weights shape: (C, K*K) -> (1, 1, C, K*K)
            weights = self.aggregation_weights.unsqueeze(0).unsqueeze(0)  # (1, 1, C, K*K)
            # Each channel's K*K output is multiplied by the corresponding weight vector and summed
            out = (quantum_outputs * weights).sum(dim=-1)  # (B, num_patches, C)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")

        # Reshape spatial dimensions
        return F.fold(
            out.permute(0, 2, 1),  # (B, C, num_patches)
            output_size=(h_out, w_out),
            kernel_size=1,  # Use 1x1 conv to aggregate as we already did the aggregation
            stride=1,
            padding=0
        )  # (B, C, h_out, w_out)

    def _compute_normalization_factor(self, x: torch.Tensor) -> torch.Tensor:
        """Compute normalization factor for the output.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Normalization factor
        """
        ones = torch.ones_like(x)
        unfolded = F.unfold(ones, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        folded = F.fold(unfolded, output_size=x.shape[2:], kernel_size=self.kernel_size, stride=self.stride,
                        padding=self.padding)
        return folded


class Quanv3d(_QuanvNd):
    """3D Quantum Convolution Layer"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize classical convolution layer for 3D  
        self.classical_conv = nn.Conv3d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the 3D quantum convolution layer"""
        self._validate_input(x)
        return self._quantum_conv(x)

    def _validate_input(self, x: torch.Tensor) -> None:
        """Check if input dimensions are as expected for 3D"""
        if x.dim() != 5:
            raise ValueError(f"Expected 5D input tensor for 3D convolution, got {x.dim()}D")
        if x.size(1) != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {x.size(1)}")

    def _quantum_conv(self, x: torch.Tensor) -> torch.Tensor:
        """Implement the 3D quantum convolution operation"""
        pass
