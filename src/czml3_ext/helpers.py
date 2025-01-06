import base64
from importlib import resources as impresources
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt

from . import data
from .data import (
    BILLBOARD_SUFFIX,
    BORDER_SUFFIX,
    available_billboards,
    available_borders,
)
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


def perform_operation(
    operation: Literal["eq", "ge", "le", "g", "l"],
    v1: npt.NDArray[np.integer | np.floating],
    v2: npt.NDArray[np.integer | np.floating] | int | float,
) -> npt.NDArray[np.bool_]:
    """Perform an operation between two inputs according to the string input `operation`.

    Parameters
    ----------
    operation : Literal[&quot;eq&quot;, &quot;ge&quot;, &quot;le&quot;, &quot;g&quot;, &quot;l&quot;]
        Operation to perform
    v1 : npt.NDArray[np.integer  |  np.floating]
        Matrix on the left side of the operation
    v2 : npt.NDArray[np.integer  |  np.floating] | int | float
        Matrix or number on the right side of the operation

    Returns
    -------
    npt.NDArray[np.bool_]
        Matrix of booleans after the operation

    Raises
    ------
    ValueError
        Operation is not one of the valid options
    """
    match operation:
        case "eq":
            mask = v1 == v2
        case "ge":
            mask = v1 >= v2
        case "le":
            mask = v1 <= v2
        case "g":
            mask = v1 > v2
        case "l":
            mask = v1 < v2
        case _:
            raise ValueError(f"Invalid operation: {operation}")
    return mask  # type: ignore
