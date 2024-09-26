from typing import Union

import numpy as np
import numpy.typing as npt
from shapely.geometry import Polygon
from shapely.geometry.polygon import LinearRing

from .definitions import TNP


def make_LLA(
    coords, m_alt: Union[int, float], *, dtype=np.float32
) -> npt.NDArray[np.floating[TNP]]:
    num_coords = len(coords)
    out = np.zeros((num_coords, 3, 1), dtype=dtype)
    out[:, [1, 0], 0] = coords
    out[:, 2, 0] = m_alt
    return out


def poly2LLA(
    polygon: Polygon, m_alt: Union[int, float] = 0.0
) -> npt.NDArray[np.floating[TNP]]:
    return make_LLA(polygon.exterior.coords, m_alt)


def multipoly2LLA(
    multipolygon: Polygon, m_alt: Union[int, float] = 0.0
) -> npt.NDArray[np.floating[TNP]]:
    raise NotImplementedError


def linear_ring2LLA(
    linear_ring: LinearRing, m_alt: Union[int, float] = 0.0
) -> npt.NDArray[np.floating[TNP]]:
    return make_LLA(linear_ring.coords, m_alt)
