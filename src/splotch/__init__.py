"""__init__.py."""

from splotch.inference import run_nuts, run_svi
from splotch.registration import register
from splotch.utils import get_input_data, savagedickey

__all__ = ["get_input_data", "register", "run_nuts", "run_svi", "savagedickey"]
