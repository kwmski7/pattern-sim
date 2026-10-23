"""precipsim: learned pattern / location / scale / rotation / intensity scores for a precipitation
forecast against an observation, without having to define objects."""
from .data import FieldBank, FieldPairSynth, download_era5
from .infer import FieldScorer
from .model import RegNet

__all__ = ["FieldBank", "FieldPairSynth", "download_era5", "FieldScorer", "RegNet"]
