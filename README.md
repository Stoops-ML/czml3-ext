# czml3-ext
![PyPI - Version](https://img.shields.io/pypi/v/czml3_ext)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/czml3_ext)
![build](https://img.shields.io/github/actions/workflow/status/Stoops-ML/czml3-ext/workflow.yml)

This library is a collection of functions that outputs `list`s of `czml3.packet`s for various items, as shown in the table below. See [czml3](https://github.com/Stoops-ML/czml3) for more information about CZML properties.

| Item     | Function in `czml3_ext.packets` |
| -------- | ------------------------------- |
| Sensor   | `sensor`                        |
| Grid     | `grid`                          |
| Border   | `border`                        |
| Coverage | `coverage`                      |

## Installation
`pip install czml3-ext`

## Examples
See the example [notebook](https://github.com/Stoops-ML/czml3-ext/blob/main/examples/examples.ipynb) for a full demo of the package. Run `pip install czml3_ext[examples]` to run the examples locally.

The following code produces a CZML file with a single sensor:
```
import numpy as np
from czml3 import Document, Packet, CZML_VERSION
from czml3.properties import (
    Color,
    Ellipsoid,
    EllipsoidRadii,
    Material,
    SolidColorMaterial,
)

from czml3_ext import packets
from czml3_ext.colours import RGBA_blue, RGBA_white

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

doc = Document(
    packets=[Packet(name="simple", id="document", version=CZML_VERSION)] + sensor
)
with open("example.czml", "w") as f:
    f.write(doc.dumps())
```

This produces the following view:
![Example](https://github.com/user-attachments/assets/c48709fe-652e-480b-a69a-ffccbe7b5ec1)




## Contributing
PRs are always welcome and appreciated!

After forking the repo install the dev requirements: `pip install -e .[dev]`.
