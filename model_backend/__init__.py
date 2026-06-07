"""plm.model_backend — self-contained model backends (ported from AFramework,
AFW couplings stripped: no @resource / ResourceManager / registries / AFW Logger).
Importing this package is dependency-free; concrete backends pull their SDK
(openai / anthropic / httpx / requests) only when imported individually."""
from .model_backend import ModelBackend

__all__ = ["ModelBackend"]
