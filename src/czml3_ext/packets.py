from collections.abc import Sequence
from typing import Optional, Union
from uuid import uuid4

import numpy as np
import numpy.typing as npt
import shapely
from czml3 import Packet
from czml3.properties import (
    Color,
    Material,
    Polygon,
    Polyline,
    PositionList,
    PositionListOfLists,
    SolidColorMaterial,
)
from transforms84.helpers import DDM2RRM, RRM2DDM
from transforms84.systems import WGS84
from transforms84.transforms import (
    AER2ENU,
    ENU2ECEF,
    ECEF2geodetic,
)

from .colours import COLOUR_TYPE, RGBA
from .definitions import TNP
from .errors import DataTypeError, MismatchedInputsError, NumDimensionsError, ShapeError
from .helpers import get_border
from .shapely_helpers import linear_ring2LLA, poly2LLA


def sensor_polyline(
    ddm_LLA: Union[
        Sequence[Union[int, float, np.integer[TNP], np.floating[TNP]]],
        npt.NDArray[Union[np.floating[TNP], np.integer[TNP]]],
    ],
    deg_az_broadside: Union[
        int,
        float,
        np.floating[TNP],
        np.integer[TNP],
        Sequence[Union[int, float, np.integer[TNP], np.floating[TNP]]],
        npt.NDArray[Union[np.floating[TNP], np.integer[TNP]]],
    ],
    deg_el_broadside: Union[
        int,
        float,
        np.floating[TNP],
        np.integer[TNP],
        Sequence[Union[int, float, np.integer[TNP], np.floating[TNP]]],
        npt.NDArray[Union[np.floating[TNP], np.integer[TNP]]],
    ],
    deg_az_FOV: Union[
        int,
        float,
        np.floating[TNP],
        np.integer[TNP],
        Sequence[Union[int, float, np.integer[TNP], np.floating[TNP]]],
        npt.NDArray[Union[np.floating[TNP], np.integer[TNP]]],
    ],
    deg_el_FOV: Union[
        int,
        float,
        np.floating[TNP],
        np.integer[TNP],
        Sequence[Union[int, float, np.integer[TNP], np.floating[TNP]]],
        npt.NDArray[Union[np.floating[TNP], np.integer[TNP]]],
    ],
    m_distance_max: Union[
        int,
        float,
        np.floating[TNP],
        np.integer[TNP],
        Sequence[Union[int, float, np.integer[TNP], np.floating[TNP]]],
        npt.NDArray[Union[np.floating[TNP], np.integer[TNP]]],
    ],
    m_distance_min: Optional[
        Union[
            int,
            float,
            np.floating[TNP],
            np.integer[TNP],
            Sequence[Union[int, float, np.floating[TNP], np.integer[TNP]]],
            npt.NDArray[Union[np.integer[TNP], np.floating[TNP]]],
        ]
    ] = None,
    *,
    name: Optional[Union[str, Sequence[str]]] = None,
    description: Optional[Union[str, Sequence[str]]] = None,
    rgba: Optional[Union[COLOUR_TYPE, Sequence[COLOUR_TYPE]]] = None,
    n_arc_points: Union[int, Sequence[int]] = 100,
) -> list[Packet]:
    """Create a sensor using polylines.

    Parameters
    ----------
    ddm_LLA : Union[ Sequence[Union[int, float, np.integer[TNP], np.floating[TNP]]], npt.NDArray[Union[np.floating[TNP], np.integer[TNP]]], ]
        Location of sensor(s) in LLA [deg, deg, m] of shape (3, 1) for one sensor of (n, 3, 1) for n sensors
    deg_az_broadside : Union[ int, float, np.floating[TNP], np.integer[TNP], Sequence[Union[int, float, np.integer[TNP], np.floating[TNP]]], npt.NDArray[Union[np.floating[TNP], np.integer[TNP]]], ]
        Azimuth of sensor(s) [deg]
    deg_el_broadside : Union[ int, float, np.floating[TNP], np.integer[TNP], Sequence[Union[int, float, np.integer[TNP], np.floating[TNP]]], npt.NDArray[Union[np.floating[TNP], np.integer[TNP]]], ]
        Elevation of sensor(s) [deg]
    deg_az_FOV : Union[ int, float, np.floating[TNP], np.integer[TNP], Sequence[Union[int, float, np.integer[TNP], np.floating[TNP]]], npt.NDArray[Union[np.floating[TNP], np.integer[TNP]]], ]
        Azimuth FOV of sensor(s) [deg]
    deg_el_FOV : Union[ int, float, np.floating[TNP], np.integer[TNP], Sequence[Union[int, float, np.integer[TNP], np.floating[TNP]]], npt.NDArray[Union[np.floating[TNP], np.integer[TNP]]], ]
        Elevation FOV of sensor(s) [deg]
    m_distance_max : Union[ int, float, np.floating[TNP], np.integer[TNP], Sequence[Union[int, float, np.integer[TNP], np.floating[TNP]]], npt.NDArray[Union[np.floating[TNP], np.integer[TNP]]], ]
        Maximum range of sensor(s) [m]
    m_distance_min : Optional[ Union[ int, float, np.floating[TNP], np.integer[TNP], Sequence[Union[int, float, np.floating[TNP], np.integer[TNP]]], npt.NDArray[Union[np.integer[TNP], np.floating[TNP]]], ] ], optional
        Minimum range of sensor(s) [m], by default None
    name : Optional[Union[str, Sequence[str]]], optional
        Display name(s), by default None
    description : Optional[Union[str, Sequence[str]]], optional
        Display description(s), by default None
    rgba : Optional[Union[COLOUR_TYPE, Sequence[COLOUR_TYPE]]], optional
        Colour of polylines, by default None
    n_arc_points : int, optional
        Number of points to use to create the arc, by default 100

    Returns
    -------
    list[Packet]
        List of packets to create the sensor

    Raises
    ------
    TypeError
        _description_
    ShapeError
        _description_
    ShapeError
        _description_
    NumDimensionsError
        _description_
    DataTypeError
        _description_
    TypeError
        _description_
    TypeError
        _description_
    TypeError
        _description_
    TypeError
        _description_
    TypeError
        _description_
    TypeError
        _description_
    MismatchedInputsError
        _description_
    """

    # checks
    if isinstance(ddm_LLA, Sequence):
        ddm_LLA = np.array(ddm_LLA)
    elif not isinstance(ddm_LLA, np.ndarray):
        raise TypeError("ddm_LLA must be a numpy array or sequence")
    if ddm_LLA.ndim == 2 and ddm_LLA.shape != (3, 1):
        raise ShapeError("A single point must be of shape (3, 1)")
    elif ddm_LLA.ndim == 3 and ddm_LLA.shape[1:] != (3, 1):
        raise ShapeError("Multiple points must be of shape (n, 3, 1)")
    elif not (ddm_LLA.ndim == 2 or ddm_LLA.ndim == 3):
        raise NumDimensionsError(
            "Point(s) must either have two dimensions with shape (3, 1) or (n, 3, 1)"
        )

    # make all inputs into sequences
    if ddm_LLA.ndim == 2:
        ddm_LLA = ddm_LLA[None, :]
    if not isinstance(ddm_LLA[0, 0, 0], np.floating):
        raise DataTypeError("Point(s) array must have a floating point data type")
    if np.isscalar(deg_az_broadside):
        deg_az_broadside = np.array([deg_az_broadside])
    elif isinstance(deg_az_broadside, Sequence):
        deg_az_broadside = np.array(deg_az_broadside)
    elif not isinstance(deg_az_broadside, np.ndarray):
        raise TypeError(
            "deg_az_broadside must be an int, float, sequence or numpy array"
        )
    if np.isscalar(deg_el_broadside):
        deg_el_broadside = np.array([deg_el_broadside])
    elif isinstance(deg_el_broadside, Sequence):
        deg_el_broadside = np.array(deg_el_broadside)
    elif not isinstance(deg_el_broadside, np.ndarray):
        raise TypeError(
            "deg_el_broadside must be an int, float, sequence or numpy array"
        )
    if np.isscalar(deg_az_FOV):
        deg_az_FOV = np.array([deg_az_FOV])
    elif isinstance(deg_az_FOV, Sequence):
        deg_az_FOV = np.array(deg_az_FOV)
    elif not isinstance(deg_az_FOV, np.ndarray):
        raise TypeError("deg_az_FOV must be an int, float, sequence or numpy array")
    if np.isscalar(deg_el_FOV):
        deg_el_FOV = np.array([deg_el_FOV])
    elif isinstance(deg_el_FOV, Sequence):
        deg_el_FOV = np.array(deg_el_FOV)
    elif not isinstance(deg_el_FOV, np.ndarray):
        raise TypeError("deg_el_FOV must be an int, float, sequence or numpy array")
    if np.isscalar(m_distance_max):
        m_distance_max = np.array([m_distance_max])
    elif isinstance(m_distance_max, Sequence):
        m_distance_max = np.array(m_distance_max)
    elif not isinstance(m_distance_max, np.ndarray):
        raise TypeError("m_distance_max must be an int, float, sequence or numpy array")
    if m_distance_min is None:
        m_distance_min = np.zeros_like(m_distance_max)
    elif np.isscalar(m_distance_min):
        m_distance_min = np.array([m_distance_min])
    elif isinstance(m_distance_min, Sequence):
        m_distance_min = np.array(m_distance_min)
    elif not isinstance(m_distance_min, np.ndarray):
        raise TypeError("m_distance_min must be an int, float, sequence or numpy array")
    if isinstance(name, str):
        name = [name]
    elif name is None:
        name = ["Sensor" for _ in range(ddm_LLA.shape[0])]
    if isinstance(description, str):
        description = [description]
    elif description is None:
        description = ["Sensor" for _ in range(ddm_LLA.shape[0])]
    if rgba is None:
        rgba = [RGBA.blue for _ in range(ddm_LLA.shape[0])]
    elif isinstance(rgba[0], float):
        rgba = [rgba for _ in range(ddm_LLA.shape[0])]  # type: ignore  # TODO FIX
    if not isinstance(n_arc_points, Sequence):
        n_arc_points = [n_arc_points for _ in range(ddm_LLA.shape[0])]
    if not (
        ddm_LLA.shape[0]
        == deg_az_broadside.size
        == deg_el_broadside.size
        == deg_az_FOV.size
        == deg_el_FOV.size
        == m_distance_max.size
        == m_distance_min.size
        == len(name)
        == len(description)
        == len(rgba)
        == len(n_arc_points)
    ):
        raise MismatchedInputsError("All inputs must have same length")

    # convert to radians
    rrm_LLA = DDM2RRM(ddm_LLA)
    rad_az_broadside = np.deg2rad(deg_az_broadside)
    rad_el_broadside = np.deg2rad(deg_el_broadside)
    rad_az_FOV = np.deg2rad(deg_az_FOV)
    rad_el_FOV = np.deg2rad(deg_el_FOV)

    out: list[Packet] = []
    for i_sensor in range(rrm_LLA.shape[0]):
        for m_distance in (m_distance_min[i_sensor], m_distance_max[i_sensor]):
            if m_distance == 0:
                continue

            # azimuth broadside lines
            ddm_LLA00 = RRM2DDM(
                ECEF2geodetic(
                    ENU2ECEF(
                        rrm_LLA[i_sensor],
                        AER2ENU(
                            np.array(
                                [
                                    [
                                        rad_az_broadside[i_sensor]
                                        - rad_az_FOV[i_sensor] / 2
                                    ],
                                    [
                                        rad_el_broadside[i_sensor]
                                        - rad_el_FOV[i_sensor] / 2
                                    ],
                                    [m_distance],
                                ]
                            )
                        ),
                        WGS84.a,
                        WGS84.b,
                    ),
                    WGS84.a,
                    WGS84.b,
                )
            )
            ddm_LLA01 = RRM2DDM(
                ECEF2geodetic(
                    ENU2ECEF(
                        rrm_LLA[i_sensor],
                        AER2ENU(
                            np.array(
                                [
                                    [
                                        rad_az_broadside[i_sensor]
                                        + rad_az_FOV[i_sensor] / 2
                                    ],
                                    [
                                        rad_el_broadside[i_sensor]
                                        - rad_el_FOV[i_sensor] / 2
                                    ],
                                    [m_distance],
                                ]
                            )
                        ),
                        WGS84.a,
                        WGS84.b,
                    ),
                    WGS84.a,
                    WGS84.b,
                )
            )
            ddm_LLA11 = RRM2DDM(
                ECEF2geodetic(
                    ENU2ECEF(
                        rrm_LLA[i_sensor],
                        AER2ENU(
                            np.array(
                                [
                                    [
                                        rad_az_broadside[i_sensor]
                                        + rad_az_FOV[i_sensor] / 2
                                    ],
                                    [
                                        rad_el_broadside[i_sensor]
                                        + rad_el_FOV[i_sensor] / 2
                                    ],
                                    [m_distance],
                                ]
                            )
                        ),
                        WGS84.a,
                        WGS84.b,
                    ),
                    WGS84.a,
                    WGS84.b,
                )
            )
            ddm_LLA10 = RRM2DDM(
                ECEF2geodetic(
                    ENU2ECEF(
                        rrm_LLA[i_sensor],
                        AER2ENU(
                            np.array(
                                [
                                    [
                                        rad_az_broadside[i_sensor]
                                        - rad_az_FOV[i_sensor] / 2
                                    ],
                                    [
                                        rad_el_broadside[i_sensor]
                                        + rad_el_FOV[i_sensor] / 2
                                    ],
                                    [m_distance],
                                ]
                            )
                        ),
                        WGS84.a,
                        WGS84.b,
                    ),
                    WGS84.a,
                    WGS84.b,
                )
            )
            out.append(
                Packet(
                    id=f"sensor{i_sensor}line00-{str(uuid4())}",
                    name=name[i_sensor],
                    description=description[i_sensor],
                    polyline=Polyline(
                        positions=PositionList(
                            cartographicDegrees=[
                                ddm_LLA[i_sensor, 1, 0],
                                ddm_LLA[i_sensor, 0, 0],
                                ddm_LLA[i_sensor, 2, 0],
                                ddm_LLA00[1, 0],
                                ddm_LLA00[0, 0],
                                ddm_LLA00[2, 0],
                            ]
                        ),
                        material=Material(
                            solidColor=SolidColorMaterial(
                                color=Color(rgba=rgba[i_sensor])
                            )
                        ),
                    ),
                )
            )
            out.append(
                Packet(
                    id=f"sensor{i_sensor}line01-{str(uuid4())}",
                    name=name[i_sensor],
                    description=description[i_sensor],
                    polyline=Polyline(
                        positions=PositionList(
                            cartographicDegrees=[
                                ddm_LLA[i_sensor, 1, 0],
                                ddm_LLA[i_sensor, 0, 0],
                                ddm_LLA[i_sensor, 2, 0],
                                ddm_LLA01[1, 0],
                                ddm_LLA01[0, 0],
                                ddm_LLA01[2, 0],
                            ]
                        ),
                        material=Material(
                            solidColor=SolidColorMaterial(
                                color=Color(rgba=rgba[i_sensor])
                            )
                        ),
                    ),
                )
            )
            out.append(
                Packet(
                    id=f"sensor{i_sensor}line11-{str(uuid4())}",
                    name=name[i_sensor],
                    description=description[i_sensor],
                    polyline=Polyline(
                        positions=PositionList(
                            cartographicDegrees=[
                                ddm_LLA[i_sensor, 1, 0],
                                ddm_LLA[i_sensor, 0, 0],
                                ddm_LLA[i_sensor, 2, 0],
                                ddm_LLA11[1, 0],
                                ddm_LLA11[0, 0],
                                ddm_LLA11[2, 0],
                            ]
                        ),
                        material=Material(
                            solidColor=SolidColorMaterial(
                                color=Color(rgba=rgba[i_sensor])
                            )
                        ),
                    ),
                )
            )
            out.append(
                Packet(
                    id=f"sensor{i_sensor}line10-{str(uuid4())}",
                    name=name[i_sensor],
                    description=description[i_sensor],
                    polyline=Polyline(
                        positions=PositionList(
                            cartographicDegrees=[
                                ddm_LLA[i_sensor, 1, 0],
                                ddm_LLA[i_sensor, 0, 0],
                                ddm_LLA[i_sensor, 2, 0],
                                ddm_LLA10[1, 0],
                                ddm_LLA10[0, 0],
                                ddm_LLA10[2, 0],
                            ]
                        ),
                        material=Material(
                            solidColor=SolidColorMaterial(
                                color=Color(rgba=rgba[i_sensor])
                            )
                        ),
                    ),
                )
            )
            out.append(
                Packet(
                    id=f"sensor{i_sensor}line0010-{str(uuid4())}",
                    name=name[i_sensor],
                    description=description[i_sensor],
                    polyline=Polyline(
                        positions=PositionList(
                            cartographicDegrees=[
                                ddm_LLA00[1, 0],
                                ddm_LLA00[0, 0],
                                ddm_LLA00[2, 0],
                                ddm_LLA10[1, 0],
                                ddm_LLA10[0, 0],
                                ddm_LLA10[2, 0],
                            ]
                        ),
                        material=Material(
                            solidColor=SolidColorMaterial(
                                color=Color(rgba=rgba[i_sensor])
                            )
                        ),
                    ),
                )
            )
            out.append(
                Packet(
                    id=f"sensor{i_sensor}line0111-{str(uuid4())}",
                    name=name[i_sensor],
                    description=description[i_sensor],
                    polyline=Polyline(
                        positions=PositionList(
                            cartographicDegrees=[
                                ddm_LLA01[1, 0],
                                ddm_LLA01[0, 0],
                                ddm_LLA01[2, 0],
                                ddm_LLA11[1, 0],
                                ddm_LLA11[0, 0],
                                ddm_LLA11[2, 0],
                            ]
                        ),
                        material=Material(
                            solidColor=SolidColorMaterial(
                                color=Color(rgba=rgba[i_sensor])
                            )
                        ),
                    ),
                )
            )

            # arcs
            for rad_el in (
                rad_el_broadside[i_sensor] - rad_el_FOV[i_sensor] / 2,
                rad_el_broadside[i_sensor] + rad_el_FOV[i_sensor] / 2,
            ):
                rad_el %= np.pi
                ddm_LLA_arc = []
                for i_arc in range(n_arc_points[i_sensor] - 1):
                    rad_az0 = (
                        rad_az_broadside[i_sensor]
                        - rad_az_FOV[i_sensor] / 2
                        + rad_az_FOV[i_sensor] * i_arc / (n_arc_points[i_sensor] - 1)
                    ) % (2 * np.pi)
                    rad_az1 = (
                        rad_az_broadside[i_sensor]
                        - rad_az_FOV[i_sensor] / 2
                        + rad_az_FOV[i_sensor]
                        * (i_arc + 1)
                        / (n_arc_points[i_sensor] - 1)
                    ) % (2 * np.pi)
                    ddm_LLA_point0 = RRM2DDM(
                        ECEF2geodetic(
                            ENU2ECEF(
                                rrm_LLA[i_sensor],
                                AER2ENU(np.array([[rad_az0], [rad_el], [m_distance]])),
                                WGS84.a,
                                WGS84.b,
                            ),
                            WGS84.a,
                            WGS84.b,
                        )
                    )
                    ddm_LLA_point1 = RRM2DDM(
                        ECEF2geodetic(
                            ENU2ECEF(
                                rrm_LLA[i_sensor],
                                AER2ENU(np.array([[rad_az1], [rad_el], [m_distance]])),
                                WGS84.a,
                                WGS84.b,
                            ),
                            WGS84.a,
                            WGS84.b,
                        )
                    )
                    ddm_LLA_arc.extend(
                        [
                            ddm_LLA_point0[1, 0],
                            ddm_LLA_point0[0, 0],
                            ddm_LLA_point0[2, 0],
                            ddm_LLA_point1[1, 0],
                            ddm_LLA_point1[0, 0],
                            ddm_LLA_point1[2, 0],
                        ]
                    )
                out.append(
                    Packet(
                        id=f"sensor{i_sensor}-{rad_el}-{m_distance}-{str(uuid4())}",
                        name=name[i_sensor],
                        description=description[i_sensor],
                        polyline=Polyline(
                            positions=PositionList(cartographicDegrees=ddm_LLA_arc),
                            material=Material(
                                solidColor=SolidColorMaterial(
                                    color=Color(rgba=rgba[i_sensor])
                                )
                            ),
                        ),
                    )
                )

    return out


def sensor_polygon(
    ddm_LLA: Union[
        Sequence[Union[int, float, np.integer[TNP], np.floating[TNP]]],
        npt.NDArray[Union[np.floating[TNP], np.integer[TNP]]],
    ],
    deg_az_broadside: Union[
        int,
        float,
        np.floating[TNP],
        np.integer[TNP],
        Sequence[Union[int, float, np.integer[TNP], np.floating[TNP]]],
        npt.NDArray[Union[np.floating[TNP], np.integer[TNP]]],
    ],
    deg_el_broadside: Union[
        int,
        float,
        np.floating[TNP],
        np.integer[TNP],
        Sequence[Union[int, float, np.integer[TNP], np.floating[TNP]]],
        npt.NDArray[Union[np.floating[TNP], np.integer[TNP]]],
    ],
    deg_az_FOV: Union[
        int,
        float,
        np.floating[TNP],
        np.integer[TNP],
        Sequence[Union[int, float, np.integer[TNP], np.floating[TNP]]],
        npt.NDArray[Union[np.floating[TNP], np.integer[TNP]]],
    ],
    deg_el_FOV: Union[
        int,
        float,
        np.floating[TNP],
        np.integer[TNP],
        Sequence[Union[int, float, np.integer[TNP], np.floating[TNP]]],
        npt.NDArray[Union[np.floating[TNP], np.integer[TNP]]],
    ],
    m_distance_max: Union[
        int,
        float,
        np.floating[TNP],
        np.integer[TNP],
        Sequence[Union[int, float, np.integer[TNP], np.floating[TNP]]],
        npt.NDArray[Union[np.floating[TNP], np.integer[TNP]]],
    ],
    m_distance_min: Optional[
        Union[
            int,
            float,
            np.floating[TNP],
            np.integer[TNP],
            Sequence[Union[int, float, np.floating[TNP], np.integer[TNP]]],
            npt.NDArray[Union[np.integer[TNP], np.floating[TNP]]],
        ]
    ] = None,
    *,
    name: Optional[Union[str, Sequence[str]]] = None,
    description: Optional[Union[str, Sequence[str]]] = None,
    rgba: Optional[Union[COLOUR_TYPE, Sequence[COLOUR_TYPE]]] = None,
    n_arc_points: Union[int, Sequence[int]] = 100,
) -> list[Packet]:
    """Create a sensor using polygons.

    Parameters
    ----------
    ddm_LLA : Union[ Sequence[Union[int, float, np.integer[TNP], np.floating[TNP]]], npt.NDArray[Union[np.floating[TNP], np.integer[TNP]]], ]
        Location of sensor(s) in LLA [deg, deg, m] of shape (3, 1) for one sensor of (n, 3, 1) for n sensors
    deg_az_broadside : Union[ int, float, np.floating[TNP], np.integer[TNP], Sequence[Union[int, float, np.integer[TNP], np.floating[TNP]]], npt.NDArray[Union[np.floating[TNP], np.integer[TNP]]], ]
        Azimuth of sensor(s) [deg]
    deg_el_broadside : Union[ int, float, np.floating[TNP], np.integer[TNP], Sequence[Union[int, float, np.integer[TNP], np.floating[TNP]]], npt.NDArray[Union[np.floating[TNP], np.integer[TNP]]], ]
        Elevation of sensor(s) [deg]
    deg_az_FOV : Union[ int, float, np.floating[TNP], np.integer[TNP], Sequence[Union[int, float, np.integer[TNP], np.floating[TNP]]], npt.NDArray[Union[np.floating[TNP], np.integer[TNP]]], ]
        Azimuth FOV of sensor(s) [deg]
    deg_el_FOV : Union[ int, float, np.floating[TNP], np.integer[TNP], Sequence[Union[int, float, np.integer[TNP], np.floating[TNP]]], npt.NDArray[Union[np.floating[TNP], np.integer[TNP]]], ]
        Elevation FOV of sensor(s) [deg]
    m_distance_max : Union[ int, float, np.floating[TNP], np.integer[TNP], Sequence[Union[int, float, np.integer[TNP], np.floating[TNP]]], npt.NDArray[Union[np.floating[TNP], np.integer[TNP]]], ]
        Maximum range of sensor(s) [m]
    m_distance_min : Optional[ Union[ int, float, np.floating[TNP], np.integer[TNP], Sequence[Union[int, float, np.floating[TNP], np.integer[TNP]]], npt.NDArray[Union[np.integer[TNP], np.floating[TNP]]], ] ], optional
        Minimum range of sensor(s) [m], by default None
    name : Optional[Union[str, Sequence[str]]], optional
        Display name(s), by default None
    description : Optional[Union[str, Sequence[str]]], optional
        Display description(s), by default None
    rgba : Optional[Union[COLOUR_TYPE, Sequence[COLOUR_TYPE]]], optional
        Colour of polygons, by default None
    n_arc_points : int, optional
        Number of points to use to create the arc, by default 100

    Returns
    -------
    list[Packet]
        List of packets to create the sensor

    Raises
    ------
    TypeError
        _description_
    ShapeError
        _description_
    ShapeError
        _description_
    NumDimensionsError
        _description_
    DataTypeError
        _description_
    TypeError
        _description_
    TypeError
        _description_
    TypeError
        _description_
    TypeError
        _description_
    TypeError
        _description_
    TypeError
        _description_
    MismatchedInputsError
        _description_
    """

    # checks
    if isinstance(ddm_LLA, Sequence):
        ddm_LLA = np.array(ddm_LLA)
    elif not isinstance(ddm_LLA, np.ndarray):
        raise TypeError("ddm_LLA must be a numpy array or sequence")
    if ddm_LLA.ndim == 2 and ddm_LLA.shape != (3, 1):
        raise ShapeError("A single point must be of shape (3, 1)")
    elif ddm_LLA.ndim == 3 and ddm_LLA.shape[1:] != (3, 1):
        raise ShapeError("Multiple points must be of shape (n, 3, 1)")
    elif not (ddm_LLA.ndim == 2 or ddm_LLA.ndim == 3):
        raise NumDimensionsError(
            "Point(s) must either have two dimensions with shape (3, 1) or (n, 3, 1)"
        )

    # make all inputs into sequences
    if ddm_LLA.ndim == 2:
        ddm_LLA = ddm_LLA[None, :]
    if not isinstance(ddm_LLA[0, 0, 0], np.floating):
        raise DataTypeError("Point(s) array must have a floating point data type")
    if np.isscalar(deg_az_broadside):
        deg_az_broadside = np.array([deg_az_broadside])
    elif isinstance(deg_az_broadside, Sequence):
        deg_az_broadside = np.array(deg_az_broadside)
    elif not isinstance(deg_az_broadside, np.ndarray):
        raise TypeError(
            "deg_az_broadside must be an int, float, sequence or numpy array"
        )
    if np.isscalar(deg_el_broadside):
        deg_el_broadside = np.array([deg_el_broadside])
    elif isinstance(deg_el_broadside, Sequence):
        deg_el_broadside = np.array(deg_el_broadside)
    elif not isinstance(deg_el_broadside, np.ndarray):
        raise TypeError(
            "deg_el_broadside must be an int, float, sequence or numpy array"
        )
    if np.isscalar(deg_az_FOV):
        deg_az_FOV = np.array([deg_az_FOV])
    elif isinstance(deg_az_FOV, Sequence):
        deg_az_FOV = np.array(deg_az_FOV)
    elif not isinstance(deg_az_FOV, np.ndarray):
        raise TypeError("deg_az_FOV must be an int, float, sequence or numpy array")
    if np.isscalar(deg_el_FOV):
        deg_el_FOV = np.array([deg_el_FOV])
    elif isinstance(deg_el_FOV, Sequence):
        deg_el_FOV = np.array(deg_el_FOV)
    elif not isinstance(deg_el_FOV, np.ndarray):
        raise TypeError("deg_el_FOV must be an int, float, sequence or numpy array")
    if np.isscalar(m_distance_max):
        m_distance_max = np.array([m_distance_max])
    elif isinstance(m_distance_max, Sequence):
        m_distance_max = np.array(m_distance_max)
    elif not isinstance(m_distance_max, np.ndarray):
        raise TypeError("m_distance_max must be an int, float, sequence or numpy array")
    if m_distance_min is None:
        m_distance_min = np.zeros_like(m_distance_max)
    elif np.isscalar(m_distance_min):
        m_distance_min = np.array([m_distance_min])
    elif isinstance(m_distance_min, Sequence):
        m_distance_min = np.array(m_distance_min)
    elif not isinstance(m_distance_min, np.ndarray):
        raise TypeError("m_distance_min must be an int, float, sequence or numpy array")
    if isinstance(name, str):
        name = [name]
    elif name is None:
        name = ["Sensor" for _ in range(ddm_LLA.shape[0])]
    if isinstance(description, str):
        description = [description]
    elif description is None:
        description = ["Sensor" for _ in range(ddm_LLA.shape[0])]
    if rgba is None:
        c = RGBA.blue.copy()
        c[3] = 100
        rgba = [c for _ in range(ddm_LLA.shape[0])]
    elif not isinstance(rgba[0], Sequence):
        rgba = [rgba for _ in range(ddm_LLA.shape[0])]  # type: ignore  # TODO FIX
    if not isinstance(n_arc_points, Sequence):
        n_arc_points = [n_arc_points for _ in range(ddm_LLA.shape[0])]
    if not (
        ddm_LLA.shape[0]
        == deg_az_broadside.size
        == deg_el_broadside.size
        == deg_az_FOV.size
        == deg_el_FOV.size
        == m_distance_max.size
        == m_distance_min.size
        == len(name)
        == len(description)
        == len(rgba)
        == len(n_arc_points)
    ):
        raise MismatchedInputsError("All inputs must have same length")

    # convert to radians
    rrm_LLA = DDM2RRM(ddm_LLA)
    rad_az_broadside = np.deg2rad(deg_az_broadside)
    rad_el_broadside = np.deg2rad(deg_el_broadside)
    rad_az_FOV = np.deg2rad(deg_az_FOV)
    rad_el_FOV = np.deg2rad(deg_el_FOV)

    out: list[Packet] = []
    for i_sensor in range(rrm_LLA.shape[0]):
        ddm_LLA00_min = RRM2DDM(
            ECEF2geodetic(
                ENU2ECEF(
                    rrm_LLA[i_sensor],
                    AER2ENU(
                        np.array(
                            [
                                [rad_az_broadside[i_sensor] - rad_az_FOV[i_sensor] / 2],
                                [rad_el_broadside[i_sensor] - rad_el_FOV[i_sensor] / 2],
                                [m_distance_min[i_sensor]],
                            ]
                        )
                    ),
                    WGS84.a,
                    WGS84.b,
                ),
                WGS84.a,
                WGS84.b,
            )
        )
        ddm_LLA01_min = RRM2DDM(
            ECEF2geodetic(
                ENU2ECEF(
                    rrm_LLA[i_sensor],
                    AER2ENU(
                        np.array(
                            [
                                [rad_az_broadside[i_sensor] + rad_az_FOV[i_sensor] / 2],
                                [rad_el_broadside[i_sensor] - rad_el_FOV[i_sensor] / 2],
                                [m_distance_min[i_sensor]],
                            ]
                        )
                    ),
                    WGS84.a,
                    WGS84.b,
                ),
                WGS84.a,
                WGS84.b,
            )
        )
        ddm_LLA11_min = RRM2DDM(
            ECEF2geodetic(
                ENU2ECEF(
                    rrm_LLA[i_sensor],
                    AER2ENU(
                        np.array(
                            [
                                [rad_az_broadside[i_sensor] + rad_az_FOV[i_sensor] / 2],
                                [rad_el_broadside[i_sensor] + rad_el_FOV[i_sensor] / 2],
                                [m_distance_min[i_sensor]],
                            ]
                        )
                    ),
                    WGS84.a,
                    WGS84.b,
                ),
                WGS84.a,
                WGS84.b,
            )
        )
        ddm_LLA10_min = RRM2DDM(
            ECEF2geodetic(
                ENU2ECEF(
                    rrm_LLA[i_sensor],
                    AER2ENU(
                        np.array(
                            [
                                [rad_az_broadside[i_sensor] - rad_az_FOV[i_sensor] / 2],
                                [rad_el_broadside[i_sensor] + rad_el_FOV[i_sensor] / 2],
                                [m_distance_min[i_sensor]],
                            ]
                        )
                    ),
                    WGS84.a,
                    WGS84.b,
                ),
                WGS84.a,
                WGS84.b,
            )
        )

        ddm_LLA00_max = RRM2DDM(
            ECEF2geodetic(
                ENU2ECEF(
                    rrm_LLA[i_sensor],
                    AER2ENU(
                        np.array(
                            [
                                [rad_az_broadside[i_sensor] - rad_az_FOV[i_sensor] / 2],
                                [rad_el_broadside[i_sensor] - rad_el_FOV[i_sensor] / 2],
                                [m_distance_max[i_sensor]],
                            ]
                        )
                    ),
                    WGS84.a,
                    WGS84.b,
                ),
                WGS84.a,
                WGS84.b,
            )
        )
        ddm_LLA01_max = RRM2DDM(
            ECEF2geodetic(
                ENU2ECEF(
                    rrm_LLA[i_sensor],
                    AER2ENU(
                        np.array(
                            [
                                [rad_az_broadside[i_sensor] + rad_az_FOV[i_sensor] / 2],
                                [rad_el_broadside[i_sensor] - rad_el_FOV[i_sensor] / 2],
                                [m_distance_max[i_sensor]],
                            ]
                        )
                    ),
                    WGS84.a,
                    WGS84.b,
                ),
                WGS84.a,
                WGS84.b,
            )
        )
        ddm_LLA11_max = RRM2DDM(
            ECEF2geodetic(
                ENU2ECEF(
                    rrm_LLA[i_sensor],
                    AER2ENU(
                        np.array(
                            [
                                [rad_az_broadside[i_sensor] + rad_az_FOV[i_sensor] / 2],
                                [rad_el_broadside[i_sensor] + rad_el_FOV[i_sensor] / 2],
                                [m_distance_max[i_sensor]],
                            ]
                        )
                    ),
                    WGS84.a,
                    WGS84.b,
                ),
                WGS84.a,
                WGS84.b,
            )
        )
        ddm_LLA10_max = RRM2DDM(
            ECEF2geodetic(
                ENU2ECEF(
                    rrm_LLA[i_sensor],
                    AER2ENU(
                        np.array(
                            [
                                [rad_az_broadside[i_sensor] - rad_az_FOV[i_sensor] / 2],
                                [rad_el_broadside[i_sensor] + rad_el_FOV[i_sensor] / 2],
                                [m_distance_max[i_sensor]],
                            ]
                        )
                    ),
                    WGS84.a,
                    WGS84.b,
                ),
                WGS84.a,
                WGS84.b,
            )
        )

        out.append(
            Packet(
                id=f"sensor{i_sensor}-{str(uuid4())}",
                name=name[i_sensor],
                description=description[i_sensor],
                polygon=Polygon(
                    perPositionHeight=True,
                    positions=PositionList(
                        cartographicDegrees=[
                            ddm_LLA00_min[1, 0],
                            ddm_LLA00_min[0, 0],
                            ddm_LLA00_min[2, 0],
                            ddm_LLA00_max[1, 0],
                            ddm_LLA00_max[0, 0],
                            ddm_LLA00_max[2, 0],
                            ddm_LLA10_max[1, 0],
                            ddm_LLA10_max[0, 0],
                            ddm_LLA10_max[2, 0],
                            ddm_LLA10_min[1, 0],
                            ddm_LLA10_min[0, 0],
                            ddm_LLA10_min[2, 0],
                        ]
                    ),
                    material=Material(
                        solidColor=SolidColorMaterial(color=Color(rgba=rgba[i_sensor]))
                    ),
                ),
            )
        )
        out.append(
            Packet(
                id=f"sensor{i_sensor}-{str(uuid4())}",
                name=name[i_sensor],
                description=description[i_sensor],
                polygon=Polygon(
                    perPositionHeight=True,
                    positions=PositionList(
                        cartographicDegrees=[
                            ddm_LLA01_min[1, 0],
                            ddm_LLA01_min[0, 0],
                            ddm_LLA01_min[2, 0],
                            ddm_LLA01_max[1, 0],
                            ddm_LLA01_max[0, 0],
                            ddm_LLA01_max[2, 0],
                            ddm_LLA11_max[1, 0],
                            ddm_LLA11_max[0, 0],
                            ddm_LLA11_max[2, 0],
                            ddm_LLA11_min[1, 0],
                            ddm_LLA11_min[0, 0],
                            ddm_LLA11_min[2, 0],
                        ]
                    ),
                    material=Material(
                        solidColor=SolidColorMaterial(color=Color(rgba=rgba[i_sensor]))
                    ),
                ),
            )
        )
        out.append(
            Packet(
                id=f"sensor{i_sensor}-{str(uuid4())}",
                name=name[i_sensor],
                description=description[i_sensor],
                polygon=Polygon(
                    perPositionHeight=True,
                    positions=PositionList(
                        cartographicDegrees=[
                            ddm_LLA01_max[1, 0],
                            ddm_LLA01_max[0, 0],
                            ddm_LLA01_max[2, 0],
                            ddm_LLA11_max[1, 0],
                            ddm_LLA11_max[0, 0],
                            ddm_LLA11_max[2, 0],
                            ddm_LLA10_max[1, 0],
                            ddm_LLA10_max[0, 0],
                            ddm_LLA10_max[2, 0],
                            ddm_LLA00_max[1, 0],
                            ddm_LLA00_max[0, 0],
                            ddm_LLA00_max[2, 0],
                        ]
                    ),
                    material=Material(
                        solidColor=SolidColorMaterial(color=Color(rgba=rgba[i_sensor]))
                    ),
                ),
            )
        )
        out.append(
            Packet(
                id=f"sensor{i_sensor}-{str(uuid4())}",
                name=name[i_sensor],
                description=description[i_sensor],
                polygon=Polygon(
                    perPositionHeight=True,
                    positions=PositionList(
                        cartographicDegrees=[
                            ddm_LLA10_min[1, 0],
                            ddm_LLA10_min[0, 0],
                            ddm_LLA10_min[2, 0],
                            ddm_LLA10_max[1, 0],
                            ddm_LLA10_max[0, 0],
                            ddm_LLA10_max[2, 0],
                            ddm_LLA11_max[1, 0],
                            ddm_LLA11_max[0, 0],
                            ddm_LLA11_max[2, 0],
                            ddm_LLA11_min[1, 0],
                            ddm_LLA11_min[0, 0],
                            ddm_LLA11_min[2, 0],
                        ]
                    ),
                    material=Material(
                        solidColor=SolidColorMaterial(color=Color(rgba=rgba[i_sensor]))
                    ),
                ),
            )
        )
        out.append(
            Packet(
                id=f"sensor{i_sensor}-{str(uuid4())}",
                name=name[i_sensor],
                description=description[i_sensor],
                polygon=Polygon(
                    perPositionHeight=True,
                    positions=PositionList(
                        cartographicDegrees=[
                            ddm_LLA00_min[1, 0],
                            ddm_LLA00_min[0, 0],
                            ddm_LLA00_min[2, 0],
                            ddm_LLA00_max[1, 0],
                            ddm_LLA00_max[0, 0],
                            ddm_LLA00_max[2, 0],
                            ddm_LLA01_max[1, 0],
                            ddm_LLA01_max[0, 0],
                            ddm_LLA01_max[2, 0],
                            ddm_LLA01_min[1, 0],
                            ddm_LLA01_min[0, 0],
                            ddm_LLA01_min[2, 0],
                        ]
                    ),
                    material=Material(
                        solidColor=SolidColorMaterial(color=Color(rgba=rgba[i_sensor]))
                    ),
                ),
            )
        )

        for m_distance in (m_distance_min[i_sensor], m_distance_max[i_sensor]):
            if m_distance == 0:
                continue

            # arcs
            ddm_LLA_arc = []
            for i, rad_el in enumerate(
                (
                    rad_el_broadside[i_sensor] - rad_el_FOV[i_sensor] / 2,
                    rad_el_broadside[i_sensor] + rad_el_FOV[i_sensor] / 2,
                )
            ):
                rad_el %= np.pi
                r = (
                    range(n_arc_points[i_sensor] - 1)
                    if i == 0
                    else range(n_arc_points[i_sensor] - 1, -2, -1)
                )
                ddm_LLA_arc_at_height = []
                for i_arc in r:
                    rad_az0 = (
                        rad_az_broadside[i_sensor]
                        - rad_az_FOV[i_sensor] / 2
                        + rad_az_FOV[i_sensor] * i_arc / (n_arc_points[i_sensor] - 1)
                    ) % (2 * np.pi)
                    rad_az1 = (
                        rad_az_broadside[i_sensor]
                        - rad_az_FOV[i_sensor] / 2
                        + rad_az_FOV[i_sensor]
                        * (i_arc + 1)
                        / (n_arc_points[i_sensor] - 1)
                    ) % (2 * np.pi)
                    ddm_LLA_point0 = RRM2DDM(
                        ECEF2geodetic(
                            ENU2ECEF(
                                rrm_LLA[i_sensor],
                                AER2ENU(np.array([[rad_az0], [rad_el], [m_distance]])),
                                WGS84.a,
                                WGS84.b,
                            ),
                            WGS84.a,
                            WGS84.b,
                        )
                    )
                    ddm_LLA_point1 = RRM2DDM(
                        ECEF2geodetic(
                            ENU2ECEF(
                                rrm_LLA[i_sensor],
                                AER2ENU(np.array([[rad_az1], [rad_el], [m_distance]])),
                                WGS84.a,
                                WGS84.b,
                            ),
                            WGS84.a,
                            WGS84.b,
                        )
                    )
                    ddm_LLA_arc.extend(
                        [
                            ddm_LLA_point0[1, 0],
                            ddm_LLA_point0[0, 0],
                            ddm_LLA_point0[2, 0],
                            ddm_LLA_point1[1, 0],
                            ddm_LLA_point1[0, 0],
                            ddm_LLA_point1[2, 0],
                        ]
                    )
                    ddm_LLA_arc_at_height.extend(
                        [
                            ddm_LLA_point0[1, 0],
                            ddm_LLA_point0[0, 0],
                            ddm_LLA_point0[2, 0],
                            ddm_LLA_point1[1, 0],
                            ddm_LLA_point1[0, 0],
                            ddm_LLA_point1[2, 0],
                        ]
                    )
                out.append(
                    Packet(
                        id=f"sensor{i_sensor}-{str(uuid4())}",
                        name=name[i_sensor],
                        description=description[i_sensor],
                        polygon=Polygon(
                            perPositionHeight=True,
                            positions=PositionList(
                                cartographicDegrees=ddm_LLA_arc_at_height
                            ),
                            material=Material(
                                solidColor=SolidColorMaterial(
                                    color=Color(rgba=rgba[i_sensor])
                                )
                            ),
                        ),
                    )
                )
            out.append(
                Packet(
                    id=f"sensor{i_sensor}-{str(uuid4())}",
                    name=name[i_sensor],
                    description=description[i_sensor],
                    polygon=Polygon(
                        perPositionHeight=True,
                        positions=PositionList(cartographicDegrees=ddm_LLA_arc),
                        material=Material(
                            solidColor=SolidColorMaterial(
                                color=Color(rgba=rgba[i_sensor])
                            )
                        ),
                    ),
                )
            )

    return out


def grid(
    ddm_LLA: Union[
        npt.NDArray[Union[np.integer[TNP], np.floating[TNP]]],
        Sequence[Union[int, float, np.floating[TNP], np.integer[TNP]]],
    ],
    rgba: Optional[
        Union[
            npt.NDArray[Union[np.integer[TNP], np.floating[TNP]]],
            Sequence[COLOUR_TYPE],
        ]
    ] = None,
    *,
    description: Optional[Union[str, Sequence[str]]] = None,
) -> list[Packet]:
    """Make a grid in CZML.

    The coordinates entered are the centre points of the grid.
    64 bit floats are recommended if the grid has high resolution.
    To support non-contiguous grids it is assumed that the resolution of the grid (in longitude and latitude) is the
    smallest difference between points.

    :param ddm_LLA: 3D numpy array containing lat [deg], long [deg], alt [m] points
    :param rgba: rgba of all grid points
    """
    # checks
    if isinstance(ddm_LLA, Sequence):
        ddm_LLA = np.array(ddm_LLA)
    elif not isinstance(ddm_LLA, np.ndarray):
        raise TypeError("ddm_LLA must be a sequence of numpy array.")
    if ddm_LLA.ndim != 3:
        raise NumDimensionsError(
            "Point(s) must either have three dimensions with shape (n, 3, 1)"
        )
    if ddm_LLA.shape[1:] != (3, 1):
        raise ShapeError("ddm_LLA array must have a shape of (n, 3, 1)")
    ddm_LLA = ddm_LLA.copy()
    ddm_LLA[:, 2, 0] = 0
    if rgba is None:
        black = list(RGBA.black)
        black[-1] = 100.0
        rgba = np.ones((ddm_LLA.shape[0], 4)) * np.array(black)
    if isinstance(rgba, Sequence):
        rgba = np.array(rgba)
    elif not isinstance(rgba, np.ndarray):
        raise TypeError("rgba must be a sequence of numpy array.")
    if rgba.ndim != 2:
        raise NumDimensionsError(
            "RGBA(s) must either have two dimensions with shape (n, 4)"
        )
    if rgba.shape[1] != 4:
        raise ShapeError("RGBA array must have a shape of (n, 4)")
    if rgba.shape[0] != ddm_LLA.shape[0]:
        raise ShapeError("rgba input must be of same length as number of grid points")
    if description is None:
        description = ["" for _ in range(ddm_LLA.shape[0])]
    if len(description) != ddm_LLA.shape[0]:
        raise ShapeError(
            "description input must be of same length as number of grid points"
        )

    # range along latitude and longitude
    deg_deltas_lat = np.abs(ddm_LLA[:, 0, 0, np.newaxis] - ddm_LLA[:, 0, 0])
    deg_delta_lat = np.min(deg_deltas_lat[deg_deltas_lat > 0])
    deg_deltas_long = np.abs(ddm_LLA[:, 1, 0, np.newaxis] - ddm_LLA[:, 1, 0])
    deg_delta_long = np.min(deg_deltas_long[deg_deltas_long > 0])

    # build grid
    out: list[Packet] = []
    for i_centre in range(ddm_LLA.shape[0]):
        # build polygon
        ddm_LLA_polygon = [
            float(ddm_LLA[i_centre, 1, 0] - deg_delta_long / 2),
            float(ddm_LLA[i_centre, 0, 0] - deg_delta_lat / 2),
            0.0,
            float(ddm_LLA[i_centre, 1, 0] - deg_delta_long / 2),
            float(ddm_LLA[i_centre, 0, 0] + deg_delta_lat / 2),
            0.0,
            float(ddm_LLA[i_centre, 1, 0] + deg_delta_long / 2),
            float(ddm_LLA[i_centre, 0, 0] + deg_delta_lat / 2),
            0.0,
            float(ddm_LLA[i_centre, 1, 0] + deg_delta_long / 2),
            float(ddm_LLA[i_centre, 0, 0] - deg_delta_lat / 2),
            0.0,
        ]
        out.append(
            Packet(
                id=f"grid{i_centre}-{str(uuid4())}",
                name=f"Grid #{i_centre}",
                polygon=Polygon(
                    positions=PositionList(cartographicDegrees=ddm_LLA_polygon),
                    material=Material(
                        solidColor=SolidColorMaterial(
                            color=Color(rgba=rgba[i_centre].tolist())
                        )
                    ),
                    outlineColor=Color(rgba=[255, 255, 255, 255]),
                    outline=True,
                ),
                description=description[i_centre],
            )
        )
    return out


def border(
    borders: Union[
        str,
        npt.NDArray[np.floating[TNP]],
        Sequence[Union[str, npt.NDArray[np.floating[TNP]]]],
    ],
    names: Optional[Union[str, Sequence[str]]] = None,
    rgba: Union[COLOUR_TYPE, Sequence[COLOUR_TYPE]] = RGBA.white,
    step: Union[int, Sequence[int]] = 1,
) -> list[Packet]:
    """Create a CZML3 packet of a border

    Parameters
    ----------
    borders : Union[ str, npt.NDArray[np.floating[TNP]], Sequence[Union[str, npt.NDArray[np.floating[TNP]]]], ]
        The border(s) packets requested
    names : Optional[Union[str, Sequence[str]]], optional
        Name for each border, by default None
    rgba : Union[COLOUR_TYPE, Sequence[COLOUR_TYPE]], optional
        Colour of polyline, by default RGBA.white
    step : Union[int, Sequence[int]], optional
        Step of border points, by default 1

    Returns
    -------
    list[Packet]
        List of CZML3 packets.

    Raises
    ------
    TypeError
        _description_
    TypeError
        _description_
    MismatchedInputsError
        _description_
    TypeError
        _description_
    """
    if isinstance(borders, str | np.ndarray):
        borders = [borders]
    if isinstance(borders, Sequence) and not all(
        [isinstance(border, str | np.ndarray) for border in borders]
    ):
        raise TypeError("Borders must be a sequence of str or numpy arrays")
    if names is None:
        names = [border if isinstance(border, str) else "" for border in borders]
    elif isinstance(names, str):
        names = [names]
    elif isinstance(names, Sequence) and not all(
        [isinstance(name, str) for name in names]
    ):
        raise TypeError("Names must be a sequence of strings")
    if isinstance(rgba, list) and isinstance(
        rgba[0], int | float | np.integer | np.floating
    ):
        rgba = [rgba for _ in range(len(borders))]
    if isinstance(step, int):
        step = [step for _ in range(len(borders))]

    # checks
    if len(borders) != len(names) != len(rgba):
        raise MismatchedInputsError("All inputs must have same length")

    out: list[Packet] = []
    for i_border in range(len(borders)):
        b = borders[i_border]
        if isinstance(b, str):
            ddm_LLA_border = get_border(b)
        elif isinstance(borders[i_border], np.ndarray):
            ddm_LLA_border = b  # type: ignore  # TODO FIX
        else:
            raise TypeError(
                "borders must either be a str or a numpy array of shape [n, 3, 1] of lat, long, alt."
            )

        out.append(
            Packet(
                id=f"border-{names[i_border]}-{str(uuid4())}",
                name=names[i_border],
                polyline=Polyline(
                    positions=PositionList(
                        cartographicDegrees=ddm_LLA_border[:: step[i_border], [1, 0, 2]]
                        .ravel()
                        .tolist()
                    ),
                    material=Material(
                        solidColor=SolidColorMaterial(color=Color(rgba=rgba[i_border]))
                    ),
                ),
            )
        )
    return out


def coverage(
    dd_LL_coverages: Union[
        Sequence[npt.NDArray[np.floating[TNP]]], npt.NDArray[np.floating[TNP]]
    ],
    dd_LL_holes: Optional[
        Union[Sequence[npt.NDArray[np.floating[TNP]]], npt.NDArray[np.floating[TNP]]]
    ] = None,
    rgba: COLOUR_TYPE = RGBA.black,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> list[Packet]:
    """Create czml3 packets of coverage (including holes).

    :param [Sequence[npt.NDArray[np.floating[TNP]]], npt.NDArray[np.floating[TNP]]] dd_LL_coverages: Contours of coverages
    :param [Sequence[npt.NDArray[np.floating[TNP]]], npt.NDArray[np.floating[TNP]]] dd_LL_holes: Contours of holes
    :param Sequence[TNUM] rgba: _description_, defaults to [255, 255, 255, 100]
    :param Optional[str] name: Name of each packet for packet, defaults to None
    :param Optional[str] description: Description for packet, defaults to None
    :return list[Packet]: _description_
    """

    if not isinstance(dd_LL_coverages, Sequence):
        dd_LL_coverages = [dd_LL_coverages]
    if dd_LL_holes is None:
        dd_LL_holes = []
    elif not isinstance(dd_LL_holes, Sequence):
        dd_LL_holes = [dd_LL_holes]

    # get holes and coverage polygons
    polys_coverage = [shapely.Polygon(d[:, [1, 0]]) for d in dd_LL_coverages]
    polys_hole = [shapely.Polygon(d[:, [1, 0]]) for d in dd_LL_holes]

    # remove holes from coverage polygons
    for i_polygon in range(len(polys_coverage)):
        for hole in polys_hole:
            if not polys_coverage[i_polygon].intersects(hole):
                continue
            polys_coverage[i_polygon] = polys_coverage[i_polygon].difference(hole)

    # create MultiPolygon
    multipolygon_coverage_per_sensor = shapely.MultiPolygon(polys_coverage)

    # create packets
    out: list[Packet] = []
    for polygon in multipolygon_coverage_per_sensor.geoms:
        ddm_polygon: npt.NDArray[np.floating[TNP]] = poly2LLA(polygon)
        ddm_holes = [
            linear_ring2LLA(interior)[:, [1, 0, 2]].ravel().tolist()
            for interior in polygon.interiors
        ]
        out.append(
            Packet(
                id=f"coverage-{str(uuid4())}",
                name=name,
                polygon=Polygon(
                    positions=PositionList(
                        cartographicDegrees=ddm_polygon[:, [1, 0, 2]].ravel().tolist()
                    ),
                    holes=PositionListOfLists(cartographicDegrees=ddm_holes),
                    material=Material(
                        solidColor=SolidColorMaterial(color=Color(rgba=rgba))
                    ),
                    outlineColor=Color(rgba=[255, 253, 55, 255]),
                    outline=True,
                ),
                description=description,
            )
        )
    return out
