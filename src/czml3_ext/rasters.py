import pathlib
import tempfile
from collections.abc import Sequence
from typing import Literal

import numpy as np
import rasterio
from rasterio import transform
from rasterio.warp import Resampling, reproject

from .definitions import RASTER_DTYPE
from .helpers import perform_operation


def ops(
    raster_path: str | pathlib.Path,
    values: int | float | Sequence[int | float],
    operation_per_value: Literal["eq", "ge", "le", "g", "l"]
    | Sequence[Literal["eq", "ge", "le", "g", "l"]] = "eq",
    *,
    out_path: str | pathlib.Path | None = None,
    band: int = 1,
    overwrite_file: bool = True,
    error_if_no_data: bool = True,
) -> pathlib.Path:
    """Create a raster representing the result of multiple operations on a raster.

    Parameters
    ----------
    raster_path : str | pathlib.Path
        Path to the raster.
    values : int | float | Sequence[int  |  float]
        Target numbers of each operation.
    operation_per_value : Literal[&quot;eq&quot;, &quot;ge&quot;, &quot;le&quot;, &quot;g&quot;, &quot;l&quot;] | Sequence[Literal[&quot;eq&quot;, &quot;ge&quot;, &quot;le&quot;, &quot;g&quot;, &quot;l&quot;]], optional
        Operation to perform on each value, by default "eq"
    out_path : str | pathlib.Path, optional
        Output raster path, by default "tmp.tif"
    band : int, optional
        Band of geotiff, by default 1
    overwrite_file : bool, optional
        Overwrite the output file if True else raise an error if the file exists, by default True
    error_if_no_data : bool, optional
        Raise an error if no data is found in the raster, by default True

    Returns
    -------
    pathlib.Path
        Raster output file path.

    Raises
    ------
    FileExistsError
        If the output file already exists and `overwrite_file` is False.
    ValueError
        If the number of rasters, target values, operations and bands are not the same.
    """
    # init
    if out_path is None:
        out_path = pathlib.Path(tempfile.mktemp(suffix=".tif"))
    if isinstance(out_path, str):
        out_path = pathlib.Path(out_path)
    if out_path.exists() and overwrite_file:
        out_path.unlink()
    elif out_path.exists():
        raise FileExistsError(f"File {out_path} already exists.")
    if not isinstance(values, Sequence):
        values = [values]
    if isinstance(operation_per_value, str):
        operation_per_value = [operation_per_value] * len(values)

    # checks
    if len(values) != len(operation_per_value):
        raise ValueError("The number of values and operations must be the same.")

    # perform operations
    with rasterio.open(raster_path) as src:
        data = src.read(band)
        mask = np.ones(data.shape, dtype=RASTER_DTYPE)
        for v, operation in zip(values, operation_per_value, strict=False):
            mask &= perform_operation(operation, data, v)
        out_meta = src.meta.copy()

    # create raster
    if error_if_no_data and not np.any(mask):
        raise ValueError("Created raster is empty.")
    out_meta.update(dtype=RASTER_DTYPE, nodata=None, count=1)
    with rasterio.open(out_path, "w", **out_meta) as dst:
        dst.write(mask.astype(RASTER_DTYPE), 1)
    return out_path


def coverage_amount(
    raster_paths: Sequence[str | pathlib.Path],
    target_values_per_raster: int | Sequence[int],
    *,
    out_path: str | pathlib.Path | None = None,
    operation_per_raster: Literal["eq", "ge", "le", "g", "l"]
    | Sequence[Literal["eq", "ge", "le", "g", "l"]] = "eq",
    delta_x: float | None = None,
    delta_y: float | None = None,
    resampling_method: Resampling = Resampling.nearest,
    band_per_raster: int | Sequence[int] = 1,
    overwrite_file: bool = True,
) -> tuple[pathlib.Path, list[int]]:
    """Create a raster representing how many times each pixel is covered by the target values from all given rasters.

    Parameters
    ----------
    raster_paths : Sequence[str  |  pathlib.Path]
        Paths to rasters.
    target_values_per_raster : int | Sequence[int]
        Values that represent coverage in each raster. If a single int is given, it is assumed that all rasters have the same target values. If a list of ints is given, it is assumed that each raster has its own target values. If a list of lists of ints is given, it is assumed that each raster has multiple target values.
    out_path : str | pathlib.Path, optional
        Raster output file path, by default "coverage.tif"
    operation_per_value : Literal[&quot;eq&quot;, &quot;ge&quot;, &quot;le&quot;, &quot;g&quot;, &quot;l&quot;] | Sequence[Literal[&quot;eq&quot;, &quot;ge&quot;, &quot;le&quot;, &quot;g&quot;, &quot;l&quot;]], optional
        Operation to perform on each value, by default "eq"
    delta_x : float | None, optional
        Pixel size along x axis, by default None
    delta_y : float | None, optional
        Pixel size along y axis, by default None
    resampling_method : Resampling, optional
        Resampling method that is passed to `reproject` method, by default Resampling.nearest
    band_per_raster : int | Sequence[int], optional
        The band of each raster to be read, by default 1
    overwrite_file : bool, optional
        Overwrite the output file if True else raise an error if the file exists, by default True

    Returns
    -------
    pathlib.Path
        Raster output file path.

    Raises
    ------
    FileExistsError
        If the output file already exists and `overwrite_file` is False.
    ValueError
        If the number of rasters, target values, operations and bands are not the same.
    """

    # init
    if out_path is None:
        out_path = pathlib.Path(tempfile.mktemp(suffix=".tif"))
    if isinstance(out_path, str):
        out_path = pathlib.Path(out_path)
    if out_path.exists() and overwrite_file:
        out_path.unlink()
    elif out_path.exists():
        raise FileExistsError(f"File {out_path} already exists.")
    if isinstance(target_values_per_raster, int):
        target_values_per_raster = [target_values_per_raster] * len(raster_paths)
    if isinstance(operation_per_raster, str):
        operation_per_raster = [operation_per_raster] * len(raster_paths)
    if not isinstance(band_per_raster, Sequence):
        band_per_raster = [band_per_raster] * len(raster_paths)

    # checks
    if not (
        len(raster_paths)
        == len(target_values_per_raster)
        == len(operation_per_raster)
        == len(band_per_raster)
    ):
        raise ValueError(
            "The number of rasters, target values, operations and bands must be the same."
        )

    # define extent
    min_x, min_y, max_x, max_y = None, None, None, None
    for f in raster_paths:
        with rasterio.open(f) as src:
            if delta_x is None:
                delta_x = src.transform.a
            else:
                assert np.isclose(delta_x, src.transform.a)
            if delta_y is None:
                delta_y = src.transform.e
            else:
                assert np.isclose(delta_y, src.transform.e)
            if min_x is None or src.bounds.left < min_x:
                min_x = src.bounds.left
            if max_x is None or src.bounds.right > max_x:
                max_x = src.bounds.right
            if min_y is None or src.bounds.bottom < min_y:
                min_y = src.bounds.bottom
            if max_y is None or src.bounds.top > max_y:
                max_y = src.bounds.top
    assert (
        min_x is not None
        and max_x is not None
        and min_y is not None
        and max_y is not None
        and delta_x is not None
        and delta_y is not None
    )
    height = int(np.ceil((max_y - min_y) / -delta_y))
    width = int(np.ceil((max_x - min_x) / delta_x))
    coverage_matrix = np.zeros((height, width), dtype=np.uint16)
    tf = transform.from_bounds(min_x, min_y, max_x, max_y, width, height)

    for f, target_value, band, operation in zip(
        raster_paths,
        target_values_per_raster,
        band_per_raster,
        operation_per_raster,
        strict=False,
    ):
        with rasterio.open(f) as src:
            resampled_data = np.zeros(coverage_matrix.shape, dtype=src.read(band).dtype)
            reproject(
                source=rasterio.band(src, band),
                destination=resampled_data,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=tf,
                dst_crs=src.crs,
                resampling=resampling_method,
            )
            coverage_matrix += perform_operation(
                operation, resampled_data, target_value
            )

    out_meta = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "int16",
        "crs": "EPSG:4326",
        "transform": tf,
    }
    with rasterio.open(out_path, "w", **out_meta) as out_raster:
        out_raster.write(coverage_matrix, 1)

    return out_path, np.unique(coverage_matrix).ravel().tolist()
