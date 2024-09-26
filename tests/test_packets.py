from importlib import resources as impresources

import numpy as np
import pytest
from czml3 import Document, Preamble

from czml3_ext.packets import grid, reset_unique_ID, sensor_polyline

from . import saved_czmls


@pytest.mark.skip(reason="Make a better test")
def test_1sensor():
    reset_unique_ID()
    sensor1 = sensor_polyline(
        np.array([[[33.0], [33], [0]]]),
        10,
        10,
        30,
        20,
        10_000,
    )
    result = Document([Preamble(name="simple")] + sensor1).dumps()
    with (impresources.files(saved_czmls) / "sensor_ddm.czml").open("r") as f:
        expected_str = f.read().strip()
    assert result == expected_str


@pytest.mark.skip(reason="Make a better test")
def test_grid():
    reset_unique_ID()
    x = np.linspace(33, 33.5, 10)
    y = np.linspace(33, 33.5, 10)
    xv, yv = np.meshgrid(x, y)
    ddm_points = []
    for i in range(x.shape[0] - 1):
        for j in range(y.shape[0] - 1):
            ddm_points.append([yv[j, i], xv[j, i], 0])
    ddm_LLA_points = np.array(ddm_points).reshape((-1, 3, 1))
    grid1 = grid(
        ddm_LLA_points,
        np.vstack(
            (
                np.linspace(0, 255, ddm_LLA_points.shape[0]),
                np.linspace(255, 0, ddm_LLA_points.shape[0]),
                np.linspace(255, 0, ddm_LLA_points.shape[0]),
                75 * np.ones((ddm_LLA_points.shape[0],)),
            )
        )
        .T.reshape((-1, 4))
        .tolist(),
    )
    result = Document([Preamble(name="simple")] + grid1).dumps()
    with (impresources.files(saved_czmls) / "grid_DDM.czml").open("r") as f:
        expected_str = f.read().strip()
    assert result == expected_str