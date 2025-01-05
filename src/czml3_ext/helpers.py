import base64
import pathlib
from collections.abc import Sequence
from importlib import resources as impresources
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
import rasterio
from rasterio import transform
from rasterio.warp import Resampling, reproject
from skimage import draw, measure

from . import data
from .data import (
    BILLBOARD_SUFFIX,
    BORDER_SUFFIX,
    available_billboards,
    available_borders,
)
from .definitions import TNP
from .errors import BillboardNotFound, BorderNotFound


def get_billboard(file_name: str | Path) -> str:
    """
    :param file_name: name of billboard to retrieve
    :return: string of base64 encoded png billboard
    """
    if isinstance(file_name, str):
        file_name = file_name.lower()
    file_name = Path(file_name)
    if file_name.suffix != BILLBOARD_SUFFIX:
        file_name = Path("".join((file_name.name, BILLBOARD_SUFFIX)))
    try:
        with (impresources.files(data) / str(file_name)).open("r") as f:
            return f.read().strip()
    except FileNotFoundError:
        raise BillboardNotFound(
            f"Billboard {file_name} not found. Available billboards: {available_billboards}"
        ) from None


def get_border(file_name: str | Path) -> npt.NDArray[np.float64]:
    """
    :param file_name: name of border file
    :return: string of czml file
    """
    if isinstance(file_name, str):
        file_name = file_name.lower()
    file_name = Path(file_name)
    if file_name.suffix != BORDER_SUFFIX:
        file_name = Path("".join((file_name.name, BORDER_SUFFIX)))
    try:
        with (impresources.files(data) / str(file_name)).open("r") as f:
            dd_LL = np.fromstring(f.read().strip(), sep=",").reshape((-1, 2))[:, [1, 0]]
        ddm_LLA = np.zeros((dd_LL.shape[0], 3, 1), dtype=np.float64)
        ddm_LLA[:, :2] = dd_LL.reshape((-1, 2, 1))
        return ddm_LLA
    except FileNotFoundError:
        raise BorderNotFound(
            f"Billboard {file_name} not found. Available billboards: {available_borders}"
        ) from None


def png2base64(file_path: str | Path) -> str:
    """
    Convert png image to billboard string for czml
    :param file_path:
    :return:
    """
    with open(file_path, "rb") as f:
        bytes_billboard = base64.b64encode(f.read())
    return "".join(("data:@file/png;base64,", bytes_billboard.decode()))


def get_contours(
    raster: npt.NDArray[np.bool_] | str | pathlib.Path,
    *,
    num_coverage: int | None = None,
    deg_origin_x: float | None = None,
    deg_origin_y: float | None = None,
    deg_size_x: float | None = None,
    deg_size_y: float | None = None,
    find_contours_level: float = 0.5,
    pc_poly_certainty_required: float = 0.9,
    error_on_uncertainty: bool = True,
    uncertain_route: Literal["coverage", "holes"] = "holes",
) -> tuple[list[npt.NDArray[np.floating[TNP]]], list[npt.NDArray[np.floating[TNP]]]]:
    """Get the contours of areas of coverage and holes from an array.

    Parameters
    ----------
    arr : npt.NDArray[np.bool_] | str | pathlib.Path
        Coverage array of boolean values or file path to raster
    num_coverage : int, optional
        Number in raster that represents coverage, by default None
    deg_origin_x : float, optional
        X origin of array, by default None
    deg_size_x : float, optional
        Size of delta x of array, by default None
    deg_origin_y : float, optional
        Y origin of array, by default None
    deg_size_y : float, optional
        Size of delta y of array, by default None
    find_contours_level : float, optional
        Level of finding contours, by default 0.5
    pc_poly_certainty_required : float, optional
        Percentage required to determine that polygon is a hole or coverage, by default 0.9
    error_on_uncertainty : bool, optional
        Raise error if identification of polygon is below pc_poly_certainty_required threshold, by default True
    uncertain_route : Literal["coverage", "holes"], optional
        If error_on_uncertainty is False then polygon will be added to either coverage or holes list, by default "holes"

    Returns
    -------
    tuple[list[npt.NDArray[np.floating[TNP]]], list[npt.NDArray[np.floating[TNP]]]]
        A tuple where the first variable is a list of coverage polygons and the second variable is a list of hole polygons

    Raises
    ------
    TypeError
        _description_
    ValueError
        _description_
    ValueError
        _description_
    """
    # init
    if isinstance(raster, str | pathlib.Path):
        if num_coverage is None:
            raise ValueError("num_coverage must be provided if raster is a file path")
        with rasterio.open(raster) as src:
            raster = src.read(1) == num_coverage
            deg_origin_x, deg_origin_y = src.bounds.left, src.bounds.top
            deg_size_x = src.transform[0]
            deg_size_y = src.transform[4]
    else:
        if deg_origin_x is None:
            raise ValueError("deg_origin_x must be provided if raster is an array")
        if deg_size_x is None:
            raise ValueError("deg_size_x must be provided if raster is an array")
        if deg_origin_y is None:
            raise ValueError("deg_origin_y must be provided if raster is an array")
        if deg_size_y is None:
            raise ValueError("deg_size_y must be provided if raster is an array")

    # checks
    if not np.issubdtype(raster.dtype, np.bool_):
        raise TypeError("Array must be of type bool")

    # get contours
    contours = measure.find_contours(raster, find_contours_level)
    dd_LL_contours = [np.zeros(c.shape, dtype=c.dtype) for c in contours]
    for i_contour, contour in enumerate(contours):
        dd_LL_contours[i_contour][:, 0] = deg_origin_y + contour[:, 0] * deg_size_y
        dd_LL_contours[i_contour][:, 1] = deg_origin_x + contour[:, 1] * deg_size_x

    # get holes and coverage polygons
    dd_LL_coverages: list[npt.NDArray[np.floating[TNP]]] = []
    dd_LL_holes: list[npt.NDArray[np.floating[TNP]]] = []
    for contour, dd_LL_contour in zip(contours, dd_LL_contours, strict=False):
        rr, cc = draw.polygon(contour[:, 0], contour[:, 1])
        if rr.size == 0 or cc.size == 0:
            raise ValueError("Contour has no size")
        pc_certainty_coverage = np.sum(raster[rr, cc]) / raster[rr, cc].size
        if pc_certainty_coverage >= pc_poly_certainty_required:
            dd_LL_coverages.append(dd_LL_contour)
        elif (1 - pc_certainty_coverage) >= pc_poly_certainty_required:
            dd_LL_holes.append(dd_LL_contour)
        elif not error_on_uncertainty and uncertain_route == "coverage":
            dd_LL_coverages.append(dd_LL_contour)
        elif not error_on_uncertainty and uncertain_route == "holes":
            dd_LL_holes.append(dd_LL_contour)
        else:
            raise ValueError(
                f"Unsure if polygon is hole or coverage. Certainty of coverage = {pc_certainty_coverage:.2f} < {pc_poly_certainty_required:.2f}, certainty of hole = {1 - pc_certainty_coverage:.2f} < {pc_poly_certainty_required:.2f}"
            )
    return dd_LL_coverages, dd_LL_holes
