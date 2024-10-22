# czml3-ext
![PyPI - Version](https://img.shields.io/pypi/v/czml3_ext)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/czml3_ext)
![PyPI - License](https://img.shields.io/pypi/l/czml3_ext)

This library is a collection of functions that outputs `list`s of `czml3.packet`s for various items, as shown in the table below. See [CZML3](https://github.com/poliastro/czml3) for more information about CZML properties.

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
from czml3 import Document, Preamble
from czml3.properties import Color, Material, Polygon, SolidColorMaterial, Ellipsoid

from czml3_ext import packets
from czml3_ext.colours import RGBA

blue = RGBA.blue.copy()
blue[-1] = 100

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
        radii=[],
        material=Material(solidColor=SolidColorMaterial(color=Color(rgba=blue))),
        outlineColor=Color(rgba=RGBA.white),
        fill=True,
        outline=True,
    ),
)

doc = Document([Preamble(name="simple")] + sensor)
with open("example.czml", "w") as f:
    doc.dump(f)
```

This produces the following view:
![Example](https://github.com/user-attachments/assets/c48709fe-652e-480b-a69a-ffccbe7b5ec1)




## Contributing
PRs are always welcome and appreciated!

After forking the repo install the dev requirements: `pip install -e .[dev]`.
