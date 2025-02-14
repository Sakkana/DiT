import torch
from torch import nn
from timm.models.vision_transformer import PatchEmbed

x = torch.rand(8, 4, 32, 32)    #[N, C, H, W]   -> [8, 4, 32, 32]
print(x.shape)                  
print(x)

input_size = 32
patch_size = 2
in_channels = 4
hidden_size = 2
patcher = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)

x = patcher(x)                  # [N, T, D] T = H * W / patch_size ** 2 -> [8, 256, 1152]
print(x.shape)
print(x)

c = torch.rand(x.shape[0], x.shape[2])
print(c.shape)                  # [N, D]
print(c)

# DitBlock

adaLN_modulation = nn.Sequential(
    # Swish， x * sigmoid(x)
    nn.SiLU(),
    nn.Linear(hidden_size, 6 * hidden_size, bias=True)
)

tmp = adaLN_modulation(c)
print(tmp.shape)

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

shift_msa, scale_msa, gate_msa, \
            shift_mlp, scale_mlp, gate_mlp \
                = tmp.chunk(6, dim=1)
print(shift_msa.shape, scale_msa.shape, gate_msa.shape, shift_mlp.shape, scale_mlp.shape, gate_mlp.shape)

result = modulate(norm1(x), shift_msa, scale_msa)

example = torch.rand(3, 2, 3)
x = torch.randint_like(example, 10)
y = torch.randint(0, 10, (3, 1, 3))
print(x)
print(y)
print(x * y)
