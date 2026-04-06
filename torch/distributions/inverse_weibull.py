# mypy: allow-untyped-defs

import torch
from torch import Tensor
from torch.distributions import constraints
from torch.distributions.exponential import Exponential
from torch.distributions.utils import euler_constant
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import AffineTransform, PowerTransform
from torch.distributions.utils import broadcast_all


__all__ = ["InverseWeibull"]


class InverseWeibull(TransformedDistribution):
    r"""
    Samples from an Inverse Weibull (also known as Fréchet, Frechet) distribution,
    parameterized by :attr:`concentration`, :attr:`scale` and :attr:`loc`
    where::

        X ~ Weibull(scale, concentration)
        Y = loc + 1 / X ~ InverseWeibull(concentration, 1 / scale, loc)

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = InverseWeibull(torch.tensor([1.0]), torch.tensor([1.0]), torch.tensor([0.0]))
        >>> m.sample()  # sample from an Inverse Weibull with concentration=1, scale=1, loc=0
        tensor([ 1.2345])

    Args:
        concentration (float or Tensor): Concentration/shape parameter (alpha).
        scale (float or Tensor): Scale parameter of the distribution (s).
        loc (float or Tensor): Location parameter of the distribution (m).
        validate_args (bool, optional): Whether to validate arguments. Default: None.
    """

    arg_constraints = {
        "concentration": constraints.positive,
        "loc": constraints.real,
        "scale": constraints.positive,
    }
    has_rsample = True
    base_dist: Exponential

    def __init__(
        self,
        concentration: Tensor | float,
        scale: Tensor | float,
        loc: Tensor | float,
        validate_args: bool | None = None,
    ) -> None:
        self.concentration, self.loc, self.scale = broadcast_all(
            concentration, loc, scale
        )
        base_dist = Exponential(
            torch.ones_like(self.scale), validate_args=validate_args
        )
        self._power_exponent = -self.concentration.reciprocal()
        transforms = [
            PowerTransform(exponent=self._power_exponent),
            AffineTransform(loc=self.loc, scale=self.scale),
        ]
        super().__init__(base_dist, transforms, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(InverseWeibull, _instance)
        new.concentration = self.concentration.expand(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        base_dist = self.base_dist.expand(batch_shape)
        new._power_exponent = -new.concentration.reciprocal()
        transforms = [
            PowerTransform(exponent=new._power_exponent),
            AffineTransform(loc=new.loc, scale=new.scale),
        ]
        super(InverseWeibull, new).__init__(base_dist, transforms, validate_args=False)
        new._validate_args = self._validate_args
        return new

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        return constraints.greater_than(self.loc)

    @property
    def mean(self) -> Tensor:
        # mean is inf for concentration <= 1
        result = self.loc + self.scale * torch.exp(
            torch.lgamma(1 - self.concentration.reciprocal())
        )
        return torch.where(self.concentration > 1, result, torch.inf)

    @property
    def mode(self) -> Tensor:
        return (
            self.loc
            + self.scale
            * (self.concentration / (self.concentration + 1))
            ** self.concentration.reciprocal()
        )

    @property
    def variance(self) -> Tensor:
        # variance is inf for concentration <= 2
        result = self.scale.pow(2) * (
            torch.exp(torch.lgamma(1 - 2 * self.concentration.reciprocal()))
            - torch.exp(2 * torch.lgamma(1 - self.concentration.reciprocal()))
        )
        return torch.where(self.concentration > 2, result, torch.inf)

    def entropy(self):
        return (
            1
            + torch.log(self.scale / self.concentration)
            + euler_constant * (1 + self.concentration.reciprocal())
        )
