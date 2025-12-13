import torch
from torch import nn
from .bindings import ModelInfo


class GlobalPoolingLayer(nn.Module):
  __annotations__ = {}

  def __init__(self) -> None:
    super().__init__()

  def forward(self, x_in: torch.Tensor) -> torch.Tensor:
    avg_vals = torch.mean(x_in, dim=(2, 3))
    max_vals = torch.amax(x_in, dim=(2, 3))
    x_out = torch.cat([avg_vals, max_vals], dim=1)
    return x_out


class GlobalPoolingBiasStructure(nn.Module):
  c_pool: int

  def __init__(self, c_x: int, c_g: int) -> None:
    super().__init__()

    self.global_pooling_layer = nn.Sequential(
      nn.BatchNorm2d(c_g),
      nn.ReLU(inplace=True),
      GlobalPoolingLayer(),
      nn.Linear(2 * c_g, c_x),
    )

  def forward(self, x_in: torch.Tensor, g_in: torch.Tensor) -> torch.Tensor:
    pool_out = self.global_pooling_layer(g_in)
    x_out = x_in + pool_out[:, :, None, None]
    return x_out


class GlobalPoolingResidualBlock(nn.Module):
  c_pool: int

  def __init__(self, c: int, c_pool: int) -> None:
    super().__init__()

    self.conv1 = nn.Sequential(
      nn.BatchNorm2d(c),
      nn.ReLU(inplace=True),
      nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False),
    )

    self.c_pool = c_pool
    self.global_pooling = GlobalPoolingBiasStructure(c - c_pool, c_pool)

    self.conv2 = nn.Sequential(
      nn.BatchNorm2d(c - c_pool),
      nn.ReLU(inplace=True),
      nn.Conv2d(c - c_pool, c, kernel_size=3, padding=1, bias=False),
    )

    last_bn = self.conv2[0]
    if isinstance(last_bn, nn.BatchNorm2d):
      nn.init.constant_(last_bn.weight, 0.0)

  def forward(self, x_in: torch.Tensor) -> torch.Tensor:
    pool_in = self.conv1(x_in)
    pool_x = pool_in[:, self.c_pool:, :, :]
    pool_g = pool_in[:, :self.c_pool, :, :]
    pool_out = self.global_pooling(pool_x, pool_g)

    x_out = self.conv2(pool_out)
    x_out = x_out + x_in
    return x_out


class ResidualBlock(nn.Module):
  __annotations__ = {}

  def __init__(self, c: int) -> None:
    super().__init__()

    self.block = nn.Sequential(
      nn.BatchNorm2d(c),
      nn.ReLU(inplace=True),
      nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False),
      nn.BatchNorm2d(c),
      nn.ReLU(inplace=True),
      nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False),
    )

    last_bn = self.block[3]
    if isinstance(last_bn, nn.BatchNorm2d):
      nn.init.constant_(last_bn.weight, 0.0)

  def forward(self, x_in: torch.Tensor) -> torch.Tensor:
    x_out = self.block(x_in)
    x_out = x_out + x_in
    return x_out


class Trunk(nn.Module):
  __annotations__ = {}

  def __init__(self, c_in: int, fc_in: int, n: int, n_pool: int, c: int, c_pool: int) -> None:
    super().__init__()

    self.conv = nn.Conv2d(c_in, c, kernel_size=5, padding=2, bias=False)
    self.fc = nn.Linear(fc_in, c)

    assert n % n_pool == 0
    pool_freq = n // n_pool
    blocks: list[nn.Module] = []
    for i in range(n):
      if i % pool_freq != pool_freq - 1:
        blocks.append(ResidualBlock(c))
      else:
        blocks.append(GlobalPoolingResidualBlock(c, c_pool))

    self.residual_blocks = nn.Sequential(*blocks)

    self.bn = nn.BatchNorm2d(c)
    self.relu = nn.ReLU(inplace=True)

  def forward(self, image_in: torch.Tensor, additional_in: torch.Tensor) -> torch.Tensor:
    image_out = self.conv(image_in)
    additional_out = self.fc(additional_in)
    blocks_in = image_out + additional_out[:, :, None, None]
    blocks_out = self.residual_blocks(blocks_in)
    merged_out = self.bn(blocks_out)
    merged_out = self.relu(merged_out)
    return merged_out


class PolicyHead(nn.Module):
  __annotations__ = {}

  def __init__(self, c: int, c_head: int) -> None:
    super().__init__()

    self.conv1 = nn.Conv2d(c, c_head, kernel_size=1)
    self.conv2 = nn.Conv2d(c, c_head, kernel_size=1)
    self.global_pooling = GlobalPoolingBiasStructure(c_head, c_head)
    self.final = nn.Sequential(
      nn.BatchNorm2d(c_head),
      nn.ReLU(inplace=True),
      nn.Conv2d(c_head, 1, kernel_size=1),
      nn.Flatten(),
    )

  def forward(self, x_in: torch.Tensor) -> torch.Tensor:
    pool_in_x = self.conv1(x_in)
    pool_in_g = self.conv2(x_in)
    pool_out = self.global_pooling(pool_in_x, pool_in_g)
    x_out = self.final(pool_out)
    return x_out


class ValueHead(nn.Module):
  __annotations__ = {}

  def __init__(self, c: int, c_head: int, c_val: int, c_support: int) -> None:
    super().__init__()

    self.global_pooling = nn.Sequential(
      nn.Conv2d(c, c_head, kernel_size=1),
      GlobalPoolingLayer(),
      nn.Linear(2 * c_head, c_val),
      nn.ReLU(inplace=True),
      nn.Linear(c_val, c_support),
    )

  def forward(self, x_in: torch.Tensor) -> torch.Tensor:
    x_out = self.global_pooling(x_in)
    return x_out


class PolicyValueNetwork(nn.Module):
  __annotations__ = {}

  def __init__(
    self,
    *,
    n: int = 6,
    n_pool: int = 2,
    c: int = 48,
    c_pool: int = 16,
    c_head: int = 16,
    c_val: int = 128,
    input_feature_count: int = ModelInfo.input_feature_count,
    additional_input_count: int = ModelInfo.additional_input_count,
    value_support_count: int = ModelInfo.value_support_count
  ) -> None:
    super().__init__()

    self.trunk = Trunk(
      c_in=input_feature_count,
      fc_in=additional_input_count,
      n=n,
      n_pool=n_pool,
      c=c,
      c_pool=c_pool,
    )

    self.policy_head = PolicyHead(c, c_head)
    self.value_head = ValueHead(c, c_head, c_val, c_support=value_support_count)

  def forward(self, image_in: torch.Tensor,
              additional_in: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    output = self.trunk(image_in, additional_in)
    priors = self.policy_head(output)
    value = self.value_head(output)
    return priors, value
