"""Type names shared by the images_to_model modules (array shapes are documented where the arrays are used)."""

from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]
AnyArray = NDArray[Any]
JsonDict = dict[str, Any]  # a JSON object as written to report.json / PROVENANCE.json
