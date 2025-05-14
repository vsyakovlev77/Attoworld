"""
This module will contain functions related to the processing of spectra.
"""
from .spectrum import *
__all__ = [name for name in globals() if callable(globals()[name]) and not name.startswith("_")]
