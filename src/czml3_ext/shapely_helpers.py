import numpy as np
import numpy.typing as npt
from shapely.geometry import Polygon
from shapely.geometry.polygon import LinearRing


def make_LLA(
    coords, m_alt: int | float, *, dtype=np.float32
) -> npt.NDArray[np.floating]:
    num_coords = len(coords)
    out = np.zeros((num_coords, 3, 1), dtype=dtype)
    out[:, [1, 0], 0] = coords
    out[:, 2, 0] = m_alt
    return out


def poly2LLA(polygon: Polygon, m_alt: int | float = 0.0) -> npt.NDArray[np.floating]:
    return make_LLA(polygon.exterior.coords, m_alt)


def multipoly2LLA(
    multipolygon: Polygon, m_alt: int | float = 0.0
) -> npt.NDArray[np.floating]:
    raise NotImplementedError


def linear_ring2LLA(
    linear_ring: LinearRing, m_alt: int | float = 0.0
) -> npt.NDArray[np.floating]:
    return make_LLA(linear_ring.coords, m_alt)
