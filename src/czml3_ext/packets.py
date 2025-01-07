import pathlib
from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import rasterio
import shapely
from czml3 import Packet
from czml3.properties import (
    Ellipsoid,
    EllipsoidRadii,
    Polygon,
    Polyline,
    Position,
    PositionList,
    PositionListOfLists,
)
from rasterio.features import shapes
from shapely import geometry
from shapely.ops import unary_union
from transforms84.helpers import DDM2RRM, RRM2DDM, wrap
from transforms84.systems import WGS84
from transforms84.transforms import (
    AER2ENU,
    ENU2ECEF,
    ECEF2geodetic,
)

from .definitions import STR_RASTER_DTYPE
from .errors import DataTypeError, MismatchedInputsError, NumDimensionsError, ShapeError
from .helpers import get_border
from .shapely_helpers import linear_ring2LLA, poly2LLA


def sensor(
    ddm_LLA: Sequence[int | float | np.integer | np.floating]
    | npt.NDArray[np.floating | np.integer],
    deg_az_broadside: int
    | float
    | np.floating
    | np.integer
    | Sequence[int | float | np.integer | np.floating]
    | npt.NDArray[np.floating | np.integer],
    deg_el_broadside: int
    | float
    | np.floating
    | np.integer
    | Sequence[int | float | np.integer | np.floating]
    | npt.NDArray[np.floating | np.integer],
    deg_az_FOV: int
    | float
    | np.floating
    | np.integer
    | Sequence[int | float | np.integer | np.floating]
    | npt.NDArray[np.floating | np.integer],
    deg_el_FOV: int
    | float
    | np.floating
    | np.integer
    | Sequence[int | float | np.integer | np.floating]
    | npt.NDArray[np.floating | np.integer],
    m_distance_max: int
    | float
    | np.floating
    | np.integer
    | Sequence[int | float | np.integer | np.floating]
    | npt.NDArray[np.floating | np.integer],
    m_distance_min: int
    | float
    | np.floating
    | np.integer
    | Sequence[int | float | np.floating | np.integer]
    | npt.NDArray[np.integer | np.floating]
    | None = None,
    *,
    subdivisions: int | Sequence[int] = 64,
    show_minimum_range_polyline: bool = True,
    max_ellipsoid_angle: float | int = 100.0,
    **update_packets,
) -> list[Packet]:
    """Create a sensor.

    All packets in the output may be updated using kwargs.
    If the value of the kwarg is a sequence with the length of the number of sensors then each value will be assigned to the CZML3 packet of it's corresponding sensor.
    If the value of the kwarg is not a sequence with the length of the number of sensors then the value will be assigned to the CZML3 packets of all sensors.
    The following czml3.properties.Polyline properties are ignored:
        - positions
    The following czml3.properties.Ellipsoid properties are ignored:
        - minimumClock
        - maximumClock
        - minimumCone
        - maximumCone
        - radii
        - innerRadii
        - outline

    Note that an Ellipsoid shape is used to create the sensor until the specified max_ellipsoid_angle. A greater azimuth FOV angle will create an arc using Polyline packets instead.
    However, this does not impace the FOV (fill=True property of Ellipsoid).

    Parameters
    ----------
    ddm_LLA : Sequence[int | float | np.integer | np.floating] | npt.NDArray[np.floating | np.integer]
        Location of sensor(s) in LLA [deg, deg, m] of shape (3, 1) for one sensor of (n, 3, 1) for n sensors
    deg_az_broadside : int | float | np.floating | np.integer | Sequence[int | float | np.integer | np.floating] | npt.NDArray[np.floating | np.integer]
        Azimuth of sensor(s) [deg]
    deg_el_broadside : int | float | np.floating | np.integer | Sequence[int | float | np.integer | np.floating] | npt.NDArray[np.floating | np.integer]
        Elevation of sensor(s) [deg]
    deg_az_FOV : int | float | np.floating | np.integer | Sequence[int | float | np.integer | np.floating] | npt.NDArray[np.floating | np.integer]
        Azimuth FOV of sensor(s) [deg]
    deg_el_FOV : int | float | np.floating | np.integer | Sequence[int | float | np.integer | np.floating] | npt.NDArray[np.floating | np.integer]
        Elevation FOV of sensor(s) [deg]
    m_distance_max : int | float | np.floating | np.integer | Sequence[int | float | np.integer | np.floating] | npt.NDArray[np.floating | np.integer]
        Maximum range of sensor(s) [m]
    m_distance_min : int | float | np.floating | np.integer | Sequence[int | float | np.floating | np.integer] | npt.NDArray[np.integer | np.floating] | None
        Minimum range of sensor(s) [m], by default None
    subdivisions : int, Sequence[int]
        The number of samples per azimuth and elevation arc, determining the granularity of the curvature, by default 64
    show_minimum_range_polyline : bool
        Show the minimum range polylines, by default True
    max_ellipsoid_angle : float, int
        The maximum angle to create an ellipsoid - any number greater than this will create a polyline for the azimuth and elevation arcs, by default 100.0

    Returns
    -------
    list[Packet]
        List of CZML3 packets.

    Raises
    ------
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
        ddm_LLA = np.array(ddm_LLA).reshape((-1, 3, 1))
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
    if not isinstance(subdivisions, Sequence):
        subdivisions = [subdivisions for _ in range(ddm_LLA.shape[0])]
    if not (
        ddm_LLA.shape[0]
        == deg_az_broadside.size
        == deg_el_broadside.size
        == deg_az_FOV.size
        == deg_el_FOV.size
        == m_distance_max.size
        == m_distance_min.size
        == len(subdivisions)
    ):
        raise MismatchedInputsError("All inputs must have same length")

    # modify additional inputs
    add_params_per_sensor: list[dict[str, Any]] = [{} for _ in range(ddm_LLA.shape[0])]
    add_params_per_sensor_polyline: list[dict[str, Any]] = [
        {} for _ in range(ddm_LLA.shape[0])
    ]
    add_params_per_sensor_ellipsoid: list[dict[str, Any]] = [
        {} for _ in range(ddm_LLA.shape[0])
    ]
    for k, v in update_packets.items():
        if isinstance(v, Polyline):
            v.__dict__.pop("positions", None)
            for i_sensor in range(ddm_LLA.shape[0]):
                add_params_per_sensor_polyline[i_sensor] = v.__dict__
        elif isinstance(v, Ellipsoid):
            v.__dict__.pop("minimumClock", None)
            v.__dict__.pop("maximumClock", None)
            v.__dict__.pop("minimumCone", None)
            v.__dict__.pop("maximumCone", None)
            v.__dict__.pop("radii", None)
            v.__dict__.pop("innerRadii", None)
            v.__dict__.pop("outline", None)
            for i_sensor in range(ddm_LLA.shape[0]):
                add_params_per_sensor_ellipsoid[i_sensor] = v.__dict__
        elif isinstance(v, Sequence) and len(v) == ddm_LLA.shape[0]:
            for i_sensor, v1 in enumerate(v):
                if isinstance(v1, Polyline):
                    v1.__dict__.pop("positions", None)
                    add_params_per_sensor_polyline[i_sensor] = v1.__dict__
                if isinstance(v1, Ellipsoid):
                    v1.__dict__.pop("minimumClock", None)
                    v1.__dict__.pop("maximumClock", None)
                    v1.__dict__.pop("minimumCone", None)
                    v1.__dict__.pop("maximumCone", None)
                    v1.__dict__.pop("radii", None)
                    v1.__dict__.pop("innerRadii", None)
                    v1.__dict__.pop("outline", None)
                    add_params_per_sensor_ellipsoid[i_sensor] = v1.__dict__
                else:
                    add_params_per_sensor[i_sensor][k] = v1
        else:
            for i_sensor in range(ddm_LLA.shape[0]):
                add_params_per_sensor[i_sensor][k] = v

    # convert to radians
    rrm_LLA = DDM2RRM(ddm_LLA)
    rad_az_broadside = np.deg2rad(deg_az_broadside)
    rad_el_broadside = np.deg2rad(deg_el_broadside)
    rad_az_FOV = np.deg2rad(deg_az_FOV)
    rad_el_FOV = np.deg2rad(deg_el_FOV)

    out: list[Packet] = []
    for i_sensor in range(rrm_LLA.shape[0]):
        # use Polyline packets for outline
        if max_ellipsoid_angle <= deg_az_FOV[i_sensor]:
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
                        **add_params_per_sensor_polyline[i_sensor],
                    ),
                    **add_params_per_sensor[i_sensor],
                )
            )
            out.append(
                Packet(
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
                        **add_params_per_sensor_polyline[i_sensor],
                    ),
                    **add_params_per_sensor[i_sensor],
                )
            )
            out.append(
                Packet(
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
                        **add_params_per_sensor_polyline[i_sensor],
                    ),
                    **add_params_per_sensor[i_sensor],
                )
            )
            out.append(
                Packet(
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
                        **add_params_per_sensor_polyline[i_sensor],
                    ),
                    **add_params_per_sensor[i_sensor],
                )
            )

            # create arcs
            for m_distance in (m_distance_min[i_sensor], m_distance_max[i_sensor]):
                if m_distance == 0:
                    continue

                # elevation arcs at min/max azimuths
                for rad_az in (
                    rad_az_broadside[i_sensor] - rad_az_FOV[i_sensor] / 2,
                    rad_az_broadside[i_sensor] + rad_az_FOV[i_sensor] / 2,
                ):
                    rad_az %= 2 * np.pi
                    ddm_LLA_arc = []
                    for i_arc in range(subdivisions[i_sensor]):
                        rad_el0 = wrap(
                            rad_el_broadside[i_sensor]
                            - rad_el_FOV[i_sensor] / 2
                            + rad_el_FOV[i_sensor]
                            * i_arc
                            / (subdivisions[i_sensor] - 1),
                            -np.pi,
                            np.pi,
                        )
                        ddm_LLA_point = RRM2DDM(
                            ECEF2geodetic(
                                ENU2ECEF(
                                    rrm_LLA[i_sensor],
                                    AER2ENU(
                                        np.array([[rad_az], [rad_el0], [m_distance]])
                                    ),
                                    WGS84.a,
                                    WGS84.b,
                                ),
                                WGS84.a,
                                WGS84.b,
                            )
                        )
                        ddm_LLA_arc.extend(
                            [
                                ddm_LLA_point[1, 0],
                                ddm_LLA_point[0, 0],
                                ddm_LLA_point[2, 0],
                            ]
                        )
                    out.append(
                        Packet(
                            polyline=Polyline(
                                positions=PositionList(cartographicDegrees=ddm_LLA_arc),
                                **add_params_per_sensor_polyline[i_sensor],
                            ),
                            **add_params_per_sensor[i_sensor],
                        )
                    )

                # azimuth arcs at min/max elevations
                for rad_el in (
                    rad_el_broadside[i_sensor] - rad_el_FOV[i_sensor] / 2,
                    rad_el_broadside[i_sensor] + rad_el_FOV[i_sensor] / 2,
                ):
                    rad_el = wrap(rad_el, -np.pi, np.pi)
                    ddm_LLA_arc = []
                    for i_arc in range(subdivisions[i_sensor]):
                        rad_az = (
                            rad_az_broadside[i_sensor]
                            - rad_az_FOV[i_sensor] / 2
                            + rad_az_FOV[i_sensor]
                            * i_arc
                            / (subdivisions[i_sensor] - 1)
                        ) % (2 * np.pi)
                        ddm_LLA_point = RRM2DDM(
                            ECEF2geodetic(
                                ENU2ECEF(
                                    rrm_LLA[i_sensor],
                                    AER2ENU(
                                        np.array([[rad_az], [rad_el], [m_distance]])
                                    ),
                                    WGS84.a,
                                    WGS84.b,
                                ),
                                WGS84.a,
                                WGS84.b,
                            )
                        )
                        ddm_LLA_arc.extend(
                            [
                                ddm_LLA_point[1, 0],
                                ddm_LLA_point[0, 0],
                                ddm_LLA_point[2, 0],
                            ]
                        )
                    out.append(
                        Packet(
                            polyline=Polyline(
                                positions=PositionList(cartographicDegrees=ddm_LLA_arc),
                                **add_params_per_sensor_polyline[i_sensor],
                            ),
                            **add_params_per_sensor[i_sensor],
                        )
                    )

        # lines to inner radii if using Ellipsoid packet
        elif m_distance_min[i_sensor] != 0 and show_minimum_range_polyline:
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
            out.append(
                Packet(
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
                        **add_params_per_sensor_polyline[i_sensor],
                    ),
                    **add_params_per_sensor[i_sensor],
                )
            )
            out.append(
                Packet(
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
                        **add_params_per_sensor_polyline[i_sensor],
                    ),
                    **add_params_per_sensor[i_sensor],
                )
            )
            out.append(
                Packet(
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
                        **add_params_per_sensor_polyline[i_sensor],
                    ),
                    **add_params_per_sensor[i_sensor],
                )
            )
            out.append(
                Packet(
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
                        **add_params_per_sensor_polyline[i_sensor],
                    ),
                    **add_params_per_sensor[i_sensor],
                )
            )

        # ellipsoid
        if (
            "fill" in add_params_per_sensor_ellipsoid[i_sensor]
            and add_params_per_sensor_ellipsoid[i_sensor]["fill"]
            and max_ellipsoid_angle <= deg_az_FOV[i_sensor]
        ) or max_ellipsoid_angle > deg_az_FOV[i_sensor]:
            out.append(
                Packet(
                    position=Position(
                        cartographicDegrees=[
                            float(ddm_LLA[i_sensor, 1, 0]),
                            float(ddm_LLA[i_sensor, 0, 0]),
                            float(ddm_LLA[i_sensor, 2, 0]),
                        ]
                    ),
                    ellipsoid=Ellipsoid(
                        minimumClock=np.deg2rad(
                            90 - deg_az_broadside[i_sensor] - deg_az_FOV[i_sensor] / 2
                        ),  # east -> north
                        maximumClock=np.deg2rad(
                            90 - deg_az_broadside[i_sensor] + deg_az_FOV[i_sensor] / 2
                        ),  # east -> north
                        minimumCone=np.deg2rad(
                            90 - deg_el_broadside[i_sensor] - deg_el_FOV[i_sensor] / 2
                        ),  # up -> down
                        maximumCone=np.deg2rad(
                            90 - deg_el_broadside[i_sensor] + deg_el_FOV[i_sensor] / 2
                        ),  # up -> down
                        radii=EllipsoidRadii(
                            cartesian=[
                                float(m_distance_max[i_sensor]),
                                float(m_distance_max[i_sensor]),
                                float(m_distance_max[i_sensor]),
                            ]
                        ),
                        innerRadii=EllipsoidRadii(
                            cartesian=[
                                max(float(m_distance_min[i_sensor]), 0.1),
                                max(float(m_distance_min[i_sensor]), 0.1),
                                max(float(m_distance_min[i_sensor]), 0.1),
                            ]
                        ),
                        outline=bool(max_ellipsoid_angle > deg_az_FOV[i_sensor]),
                        **add_params_per_sensor_ellipsoid[i_sensor],
                    ),
                    **add_params_per_sensor[i_sensor],
                )
            )

    return out


def grid(
    ddm_LLA: npt.NDArray[np.integer | np.floating]
    | Sequence[int | float | np.floating | np.integer],
    deg_zero_tolerance_lat: float = 10e-5,
    deg_zero_tolerance_long: float = 10e-5,
    *,
    ddm_LLA_cut: None
    | npt.NDArray[np.floating]
    | Sequence[int | float | np.floating | np.integer] = None,
    **update_packets,
) -> list[Packet]:
    """Make a grid in CZML.

    The coordinates entered are the centre points of the grid.
    64 bit floats are recommended if the grid has high resolution.
    To support non-contiguous grids it is assumed that the resolution of the grid (in longitude and latitude) is the
    smallest difference between points.

    If the polygons of the grid are extremely small then play with deg_zero_tolerance_lat and deg_zero_tolerance_long: increasing these values will find the correct minimum latittudes and longitudes (in degrees).

    All packets in the output may be updated using kwargs.
    If the value of the kwarg is a sequence with the length of the number of grid points then each value will be assigned to the CZML3 packet of it's corresponding grid point.
    If the value of the kwarg is not a sequence with the length of the number of grid points then the value will be assigned to the CZML3 packets of all grid points.
    Note that the following czml3.properties.Polygon properties are ignored:
        - positions

    Parameters
    ----------
    ddm_LLA : npt.NDArray[np.integer | np.floating] | Sequence[int | float | np.floating | np.integer]
        3D numpy array or sequence containing lat [deg], long [deg], alt [m] points
    deg_zero_tolerance_lat : float
        Tolerance of 0 degrees for latittude
    deg_zero_tolerance_long : float
        Tolerance of 0 degrees for longitude
    ddm_LLA_cut : None | npt.NDArray[np.floating] | Sequence[int | float | np.floating | np.integer]
        3D numpy array or sequence containing lat [deg], long [deg], alt [m] points that will cut the polygons.

    Returns
    -------
    list[Packet]
        List of CZML3 packets.

    Raises
    ------
    TypeError
        _description_
    NumDimensionsError
        _description_
    ShapeError
        _description_
    NumDimensionsError
        _description_
    ShapeError
        _description_
    """
    # checks
    if deg_zero_tolerance_lat < 0:
        raise ValueError("deg_zero_tolerance_lat must be equal to or larger than 0.")
    if deg_zero_tolerance_long < 0:
        raise ValueError("deg_zero_tolerance_long must be equal to or larger than 0.")
    if isinstance(ddm_LLA, Sequence):
        ddm_LLA = np.array(ddm_LLA).reshape((-1, 3, 1))
    if ddm_LLA.ndim != 3:
        raise NumDimensionsError(
            "Point(s) must have three dimensions with shape (n, 3, 1)"
        )
    if ddm_LLA.shape[1:] != (3, 1):
        raise ShapeError("ddm_LLA array must have a shape of (n, 3, 1)")
    ddm_LLA = ddm_LLA.copy()
    ddm_LLA[:, 2, 0] = 0
    if ddm_LLA_cut is not None and isinstance(ddm_LLA_cut, Sequence):
        ddm_LLA_cut = np.array(ddm_LLA_cut).reshape((-1, 3, 1))
    if ddm_LLA_cut is not None and ddm_LLA_cut.ndim != 3:
        raise NumDimensionsError(
            "Border point must have three dimensions with shape (n, 3, 1)"
        )
    if ddm_LLA_cut is not None and ddm_LLA_cut.shape[1:] != (3, 1):
        raise ShapeError("ddm_LLA_border array must have a shape of (n, 3, 1)")

    # range along latitude and longitude
    deg_deltas_lat = np.abs(ddm_LLA[:, 0, 0, np.newaxis] - ddm_LLA[:, 0, 0])
    deg_delta_lat = np.min(deg_deltas_lat[deg_deltas_lat > deg_zero_tolerance_lat])
    deg_deltas_long = np.abs(ddm_LLA[:, 1, 0, np.newaxis] - ddm_LLA[:, 1, 0])
    deg_delta_long = np.min(deg_deltas_long[deg_deltas_long > deg_zero_tolerance_long])

    # modify additional inputs
    add_params_per_square: list[dict[str, Any]] = [{} for _ in range(ddm_LLA.shape[0])]
    add_params_per_square_polygon: list[dict[str, Any]] = [
        {} for _ in range(ddm_LLA.shape[0])
    ]
    for k, v in update_packets.items():
        if isinstance(v, Polygon):
            v.__dict__.pop("positions", None)
            for i_sensor in range(ddm_LLA.shape[0]):
                add_params_per_square_polygon[i_sensor] = v.__dict__
        elif isinstance(v, Sequence) and len(v) == ddm_LLA.shape[0]:
            for i_sensor, v1 in enumerate(v):
                if isinstance(v1, Polygon):
                    v1.__dict__.pop("positions", None)
                    add_params_per_square_polygon[i_sensor] = v1.__dict__
                else:
                    add_params_per_square[i_sensor][k] = v1
        else:
            for i_sensor in range(ddm_LLA.shape[0]):
                add_params_per_square[i_sensor][k] = v

    # build grid
    if ddm_LLA_cut is not None:
        poly_border = shapely.Polygon(ddm_LLA_cut[:, :2, 0])
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

        # cut with border
        if ddm_LLA_cut is not None:
            poly_polygon = shapely.Polygon(
                np.array(ddm_LLA_polygon).T.reshape((-1, 3, 1))[:, [1, 0], 0]
            )
            if not (
                poly_border.contains(poly_polygon)
                or poly_border.intersects(poly_polygon)
            ):
                continue
            elif poly_border.intersects(poly_polygon):
                poly_intersects = poly_border.intersection(poly_polygon)
                if isinstance(poly_intersects, shapely.Polygon):
                    poly_intersects = [poly_intersects]
                elif isinstance(poly_intersects, shapely.MultiPolygon):
                    poly_intersects = list(poly_intersects.geoms)
                for poly_intersect in poly_intersects:
                    np_ddm_LLA_polygon = np.zeros(
                        (len(poly_intersect.exterior.coords.xy[0]), 3), dtype=np.float32
                    )
                    np_ddm_LLA_polygon[:, :2] = np.array(
                        poly_intersect.exterior.coords.xy
                    ).T.reshape((-1, 2))[:, [1, 0]]
                    ddm_LLA_polygon = np_ddm_LLA_polygon.ravel().tolist()
                    out.append(
                        Packet(
                            polygon=Polygon(
                                positions=PositionList(
                                    cartographicDegrees=ddm_LLA_polygon
                                ),
                                **add_params_per_square_polygon[i_centre],
                            ),
                            **add_params_per_square[i_centre],
                        )
                    )
        else:
            out.append(
                Packet(
                    polygon=Polygon(
                        positions=PositionList(cartographicDegrees=ddm_LLA_polygon),
                        **add_params_per_square_polygon[i_centre],
                    ),
                    **add_params_per_square[i_centre],
                )
            )
    return out


def border(
    borders: str | npt.NDArray[np.floating] | Sequence[str | npt.NDArray[np.floating]],
    steps: int | Sequence[int] = 1,
    **update_packets,
) -> list[Packet]:
    """Create a CZML3 packet of a border.

    All packets in the output may be updated using kwargs.
    If the value of the kwarg is a sequence with the length of the number of borders then each value will be assigned to the CZML3 packet of it's corresponding border.
    If the value of the kwarg is not a sequence with the length of the number of borders then the value will be assigned to the CZML3 packets of all borders.
    Note that the following czml3.properties.Polyline properties are ignored:
        - positions

    Parameters
    ----------
    borders : str | npt.NDArray[np.floating] | Sequence[str | npt.NDArray[np.floating]]
        The border(s) packets requested
    step : int, Sequence[int], optional
        Step of border points, by default 1

    Returns
    -------
    list[Packet]
        List of CZML3 packets.

    Raises
    ------
    TypeError
        _description_
    """
    if isinstance(borders, str | np.ndarray):
        borders = [borders]
    if isinstance(steps, int):
        steps = [steps for _ in range(len(borders))]

    # modify additional inputs
    add_params_per_border: list[dict[str, Any]] = [{} for _ in range(len(borders))]
    add_params_per_border_polyline: list[dict[str, Any]] = [
        {} for _ in range(len(borders))
    ]
    for k, v in update_packets.items():
        if isinstance(v, Polyline):
            v.__dict__.pop("positions", None)
            for i_sensor in range(len(borders)):
                add_params_per_border_polyline[i_sensor] = v.__dict__
        elif isinstance(v, Sequence) and len(v) == len(borders):
            for i_sensor, v1 in enumerate(v):
                if isinstance(v1, Polyline):
                    v1.__dict__.pop("positions", None)
                    add_params_per_border_polyline[i_sensor] = v1.__dict__
                else:
                    add_params_per_border[i_sensor][k] = v1
        else:
            for i_sensor in range(len(borders)):
                add_params_per_border[i_sensor][k] = v

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
                polyline=Polyline(
                    positions=PositionList(
                        cartographicDegrees=ddm_LLA_border[
                            :: steps[i_border], [1, 0, 2]
                        ]
                        .ravel()
                        .tolist()
                    ),
                    **add_params_per_border_polyline[i_border],
                ),
                **add_params_per_border[i_border],
            )
        )
    return out


def coverage(
    raster_paths_coverage: Sequence[str | pathlib.Path] | str | pathlib.Path,
    raster_paths_hole: Sequence[str | pathlib.Path] | str | pathlib.Path | None = None,
    band_per_raster_coverage: int | Sequence[int] = 1,
    band_per_raster_hole: int | Sequence[int] = 1,
    *,
    delete_rasters: bool = False,
    **update_packets,
) -> list[Packet]:
    """Create czml3 packets of coverage (including holes).

    The rasters must have a data type of `RASTER_DTYPE`.

    All packets in the output may be updated using kwargs.
    Each value of the kwarg will be assigned to all CZML3 packets.
    Note that the following czml3.properties.Polygon properties are ignored:
        - positions
        - holes

    Parameters
    ----------
    raster_paths_coverage : Sequence[str  |  pathlib.Path] | str | pathlib.Path
        Path(s) to the raster(s) that will be used for coverage.
    raster_paths_hole : Sequence[str  |  pathlib.Path] | str | pathlib.Path | None, optional
        Path(s) to the raster(s) that will be used for holes, by default None
    band_per_raster_coverage : int | Sequence[int], optional
        Band(s) of each coverage raster, by default 1
    band_per_raster_hole : int | Sequence[int], optional
        Band(s) of each hole raster, by default 1
    delete_rasters : bool, optional
        If True then the function will delete the inputted rasters, by default False

    Returns
    -------
    list[Packet]
        List of czml3 packets.

    Raises
    ------
    ValueError
        Raised if the number of coverage rasters must be equal to the number of provided bands..
    ValueError
        Raised if the number of hole rasters must be equal to the number of provided bands..
    ValueError
        Raised if the coverage raster is not of type `RASTER_DTYPE`.
    ValueError
        Raised if the hole raster is not of type `RASTER_DTYPE`.
    """
    # init
    if isinstance(raster_paths_coverage, str | pathlib.Path):
        raster_paths_coverage = [raster_paths_coverage]
    if not isinstance(band_per_raster_coverage, Sequence):
        band_per_raster_coverage = [band_per_raster_coverage] * len(
            raster_paths_coverage
        )
    if raster_paths_hole is None:
        raster_paths_hole = []
    if isinstance(raster_paths_hole, str | pathlib.Path):
        raster_paths_hole = [raster_paths_hole]
    if not isinstance(band_per_raster_hole, Sequence):
        band_per_raster_hole = [band_per_raster_hole] * len(raster_paths_hole)
    fpath_rasters_coverages = list(map(pathlib.Path, raster_paths_coverage))
    fpath_rasters_holes = list(map(pathlib.Path, raster_paths_hole))

    # checks
    if len(fpath_rasters_coverages) != len(band_per_raster_coverage):
        raise ValueError(
            "The number of coverage rasters must be equal to the number of provided bands."
        )
    if len(fpath_rasters_holes) != len(band_per_raster_hole):
        raise ValueError(
            "The number of hole rasters must be equal to the number of provided bands."
        )

    polys_coverage: list[shapely.Polygon] = []
    for raster_path, band in zip(
        fpath_rasters_coverages, band_per_raster_coverage, strict=False
    ):
        with rasterio.open(raster_path) as src:
            if src.dtypes[band - 1] != STR_RASTER_DTYPE:
                raise ValueError(f"Raster must be of type {STR_RASTER_DTYPE}.")
            raster_data = src.read(band)
            for geom, _ in shapes(
                raster_data, mask=raster_data, transform=src.transform
            ):
                # if np.isin(value, target_value):
                polys_coverage.append(geometry.shape(geom))

    polys_hole: list[shapely.Polygon] = []
    for raster_path, band in zip(
        fpath_rasters_holes, band_per_raster_hole, strict=False
    ):
        with rasterio.open(raster_path) as src:
            if src.dtypes[band - 1] != STR_RASTER_DTYPE:
                raise ValueError(f"Raster must be of type {STR_RASTER_DTYPE}.")
            raster_data = src.read(band)
            for geom, _ in shapes(
                raster_data, mask=raster_data, transform=src.transform
            ):
                # if np.isin(value, target_value):
                polys_hole.append(geometry.shape(geom))

    # delete raster files
    if delete_rasters:
        for paths in (fpath_rasters_coverages, fpath_rasters_holes):
            for path in paths:
                path.unlink()

    # remove holes from coverage polygons
    multipolygon_coverage_per_sensor = unary_union(polys_coverage)
    poly_holes = unary_union(polys_hole)
    multipolygon_coverage_per_sensor = multipolygon_coverage_per_sensor.difference(
        poly_holes
    )
    if isinstance(multipolygon_coverage_per_sensor, shapely.Polygon):
        multipolygon_coverage_per_sensor = shapely.MultiPolygon(
            [multipolygon_coverage_per_sensor]
        )

    # modify additional inputs
    add_params1: dict[str, Any] = {}
    add_params_polygon: dict[str, Any] = {}
    for k, v in update_packets.items():
        if isinstance(v, Polygon):
            v.__dict__.pop("positions", None)
            v.__dict__.pop("holes", None)
            add_params_polygon = v.__dict__
        else:
            add_params1[k] = v

    # create packets
    out: list[Packet] = []
    for polygon in multipolygon_coverage_per_sensor.geoms:
        ddm_polygon: npt.NDArray[np.floating] = poly2LLA(polygon)
        ddm_holes = [
            linear_ring2LLA(interior)[:, [1, 0, 2]].ravel().tolist()
            for interior in polygon.interiors
        ]
        out.append(
            Packet(
                polygon=Polygon(
                    positions=PositionList(
                        cartographicDegrees=ddm_polygon[:, [1, 0, 2]].ravel().tolist()
                    ),
                    holes=PositionListOfLists(cartographicDegrees=ddm_holes)
                    if len(ddm_holes) > 0
                    else None,
                    **add_params_polygon,
                ),
                **add_params1,
            )
        )
    return out
