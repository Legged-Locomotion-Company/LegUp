from typing import Any
from omegaconf import OmegaConf


def validate_single(klass: Any) -> None:
    OmegaConf.merge(klass, OmegaConf.structured(type(klass)))


def validate(*classes) -> None:
    [validate_single(klass) for klass in classes]
