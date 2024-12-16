import numpy as np
from czml3 import Document, Preamble
from czml3.properties import (
    Color,
    Ellipsoid,
    EllipsoidRadii,
    Material,
    SolidColorMaterial,
)

from czml3_ext import packets
from czml3_ext.colours import RGBA_blue, RGBA_white


def test_readme():
    """The example in readme.md"""
    sensor = packets.sensor(
        np.array([[31.4], [34.7], [1000]]),
        10,
        30,
        100,
        20,
        10_000,
        5_000,
        name="A Sensor",
        ellipsoid=Ellipsoid(
            radii=EllipsoidRadii(cartesian=[0, 0, 0]),
            material=Material(
                solidColor=SolidColorMaterial(
                    color=Color(rgba=RGBA_blue.get_with_temp_alpha(100))
                )
            ),
            outlineColor=Color(rgba=RGBA_white),
            fill=True,
            outline=True,
        ),
    )

    doc = Document(packets=[Preamble(name="simple")] + sensor)
    with open("example.czml", "w") as f:
        f.write(doc.dumps())
