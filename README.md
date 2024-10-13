# czml3-ext
![PyPI - Version](https://img.shields.io/pypi/v/czml3_ext)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/czml3_ext)
![PyPI - License](https://img.shields.io/pypi/l/czml3_ext)

This library is a collection of functions that outputs `list`s of `czml3.packet`s for various items, as shown in the table below. See [CZML3](https://github.com/poliastro/czml3) for more information about CZML properties.

| Item     | Function in `czml3_ext.packets`        |
| -------- | -------------------------------------- |
| Sensor   | `sensor_polyline` and `sensor_polygon` |
| Grid     | `grid`                                 |
| Border   | `border`                               |
| Coverage | `coverage`                             |

## Installation
`pip install czml3-ext`

## Examples
See the example [notebook](https://github.com/Stoops-ML/czml3-ext/blob/main/examples/examples.ipynb) for a full demo of the package. Run `pip install czml3_ext[examples]` to run the examples locally.

The following code produces a CZML file with a single sensor:
```
import numpy as np
from czml3 import Document, Preamble
from czml3.properties import Color, Material, Polygon, SolidColorMaterial

from czml3_ext import packets
from czml3_ext.colours import RGBA

blue = RGBA.blue.copy()
blue[-1] = 100

sensor_polyline = packets.sensor_polyline(
    np.array([[31.8], [34.68], [0]]), 90, 30, 50, 20, 20_000, 5_000
)
sensor_polygon = packets.sensor_polygon(
    np.array([[31.8], [34.68], [0]]),
    90,
    30,
    50,
    20,
    20_000,
    5_000,
    polygon=Polygon(
        positions=[],
        material=Material(solidColor=SolidColorMaterial(color=Color(rgba=blue))),
    ),
)

doc = Document([Preamble(name="simple")] + sensor_polygon + sensor_polyline)
with open("example.czml", "w") as f:
    doc.dump(f)
```

This produces the following view:
![Example](https://github.com/user-attachments/assets/3127699f-5235-4acd-b218-c6e43f6595d4)



## Contributing
PRs are always welcome and appreciated!

After forking the repo install the dev requirements: `pip install -e .[dev]`.
