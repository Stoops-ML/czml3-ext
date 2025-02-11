import numpy as np
from colourings import Colour
from czml3 import CZML_VERSION, Document, Packet
from czml3.properties import (
    Color,
    Ellipsoid,
    EllipsoidRadii,
    Material,
    SolidColorMaterial,
)

from czml3_ext import packets


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
                    color=Color(rgbaf=Colour("blue", alpha=0.3).rgba)
                )
            ),
            outlineColor=Color(rgba=Colour("white").rgb),
            fill=True,
            outline=True,
        ),
    )

    doc = Document(
        packets=[Packet(name="simple", id="document", version=CZML_VERSION)] + sensor
    )
    with open("example.czml", "w") as f:
        f.write(doc.dumps())
