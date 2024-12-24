import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F

from components.qkernel import QKernel

__all__ = ["_QuanvNd", "Quanv2d"]


class _QuanvNd(nn.Module):
    __constants__ = ["in_channels", "out_channels", "kernel_size", "stride", "padding", "num_qlayers", "qdevice",
                     "diff_method"]

    def __init__(self):
        super().__init__()
        pass


class Quanv2d(_QuanvNd):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 2,
                 stride: int = 1,
                 padding: int = 0,
                 qkernel: QKernel = None,
                 num_qlayers: int = 2,
                 qdevice: str = "default.qubit",
                 qdevice_kwargs: dict = None,
                 diff_method: str = "best"):
        super(Quanv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.qkernel = qkernel or QKernel(quantum_channels=in_channels, kernel_size=kernel_size, num_param_blocks=num_qlayers)
        self.qdevice_kwargs = qdevice_kwargs or {}
        self.qdevice = qml.device(qdevice, wires=self.qkernel.num_qubits, **self.qdevice_kwargs)
        self.qnode = qml.QNode(self.qkernel.circuit, device=self.qdevice, interface="torch", diff_method=diff_method)
        self.qlayer = qml.qnn.TorchLayer(self.qnode, self.qkernel.weight_shapes)

        # Use 1x1 classical convolution to match the desired output channels
        self.classical_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # Verified params
        assert len(x.shape) == 4, "Input tensor must have 4 dimensions: (batch_size, channels, height, width)"
        assert x.dtype == torch.float32, "Input tensor must have dtype torch.float32"
        assert x.shape[1] == self.in_channels, f"Input tensor must have {self.in_channels} input channels"

        # Apply quantum convolution
        x = self.quantum_conv(x)

        # Apply 1x1 classical convolution to match the desired output channels
        x = self.classical_conv(x)

        return x

    def quantum_conv(self, x):
        bs, _, h, w = x.shape

        # Apply padding to the input tensor
        if self.padding != 0:
            x = F.pad(x, (self.padding,) * 4, mode="constant", value=0)
            h += 2 * self.padding
            w += 2 * self.padding

        # Unfold the input tensor to extract overlapping patches
        patches = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        patches = patches.permute(0, 2, 1)  # Reshape to (bs, num_patches, in_channels * kernel_size^2)

        # Apply quantum kernel to x
        out = []
        # Apply quantum layer to each batch
        for i in range(bs):
            batch_out = [self.qlayer(patch) for patch in patches[i]]
            out.append(torch.stack(batch_out))
        out = torch.stack(out)

        # Fold the output tensor
        out = out.permute(0, 2, 1)
        out = F.fold(out, output_size=(h, w), kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)

        # Normalization
        ones = torch.ones_like(x)
        unfolded_ones = F.unfold(ones, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        folded_ones = F.fold(unfolded_ones, output_size=(h, w), kernel_size=self.kernel_size, stride=self.stride,
                             padding=self.padding)
        out = out / folded_ones

        return out


class HybridQuanv2d(_QuanvNd):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 2,
                 stride: int = 1,
                 padding: int = 0,
                 qkernel: QKernel = None,
                 num_qlayers: int = 2,
                 qdevice: str = "default.qubit",
                 qdevice_kwargs: dict = None,
                 diff_method: str = "best"):
        super(HybridQuanv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.qkernel = qkernel or QKernel(quantum_channels=in_channels, kernel_size=kernel_size, num_param_blocks=num_qlayers)
        self.qdevice_kwargs = qdevice_kwargs or {}
        self.qdevice = qml.device(qdevice, wires=self.qkernel.num_qubits, **self.qdevice_kwargs)
        self.qnode = qml.QNode(self.qkernel.circuit, device=self.qdevice, interface="torch", diff_method=diff_method)
        self.qlayer = qml.qnn.TorchLayer(self.qnode, self.qkernel.weight_shapes)

        # Use 1x1 classical convolution to match the desired output channels
        self.classical_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        pass
