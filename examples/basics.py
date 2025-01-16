import marimo

__generated_with = "0.10.13"
app = marimo.App()


@app.cell
def _():
    import pathlib

    import marimo as mo
    import numpy as np
    import shapely
    from czml3 import CZML_VERSION, Document, Packet
    from czml3.properties import (
        Color,
        Ellipsoid,
        EllipsoidRadii,
        Material,
        Polygon,
        Polyline,
        PolylineMaterial,
        PositionList,
        SolidColorMaterial,
    )

    import czml3_ext
    from czml3_ext import packets
    from czml3_ext.colours import (
        RGBA_blue,
        RGBA_orange,
        RGBA_white,
        create_palette,
    )
    from czml3_ext.helpers import get_border

    return (
        CZML_VERSION,
        Color,
        Document,
        Ellipsoid,
        EllipsoidRadii,
        Material,
        Packet,
        Polygon,
        Polyline,
        PolylineMaterial,
        PositionList,
        RGBA_blue,
        RGBA_orange,
        RGBA_white,
        SolidColorMaterial,
        create_palette,
        czml3_ext,
        get_border,
        mo,
        np,
        packets,
        pathlib,
        shapely,
    )


@app.cell
def _(__file__, pathlib):
    fdir = pathlib.Path(__file__).parent
    return (fdir,)


@app.cell
def _(czml3_ext):
    print(f"Using czml3-ext version: {czml3_ext.__version__}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Border
        A single border
        """
    )
    return


@app.cell
def _(packets):
    border = packets.border("Israel")
    return (border,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Multiple borders""")
    return


@app.cell
def _(packets):
    borders = packets.border(["syria", "lebanon", "jordan"])
    return (borders,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Sensors
        A single sensor
        """
    )
    return


@app.cell
def _(
    Color,
    Ellipsoid,
    EllipsoidRadii,
    Material,
    RGBA_blue,
    RGBA_white,
    SolidColorMaterial,
    np,
    packets,
):
    sensor1 = packets.sensor(
        np.array([[31.4], [34.7], [1000]]),
        10,
        30,
        100,
        20,
        10_000,
        5000,
        name="A Sensor",  # kwargs add to the CZML3 packet
        ellipsoid=Ellipsoid(
            radii=EllipsoidRadii(
                cartesian=[0, 0, 0]
            ),  # required to create an Ellipsoid(), and is ignored by czml3-ext
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
    return (sensor1,)


@app.cell
def _(
    Color,
    Ellipsoid,
    EllipsoidRadii,
    Material,
    RGBA_blue,
    SolidColorMaterial,
    np,
    packets,
):
    sensor2 = packets.sensor(
        np.array([[31.4], [34.9], [1000]]),
        10,
        30,
        300,
        20,
        10_000,
        5000,
        name="A Large Sensor",  # kwargs add to the CZML3 packet
        ellipsoid=Ellipsoid(
            radii=EllipsoidRadii(
                cartesian=[0, 0, 0]
            ),  # required to create an Ellipsoid(), and is ignored by czml3-ext
            material=Material(
                solidColor=SolidColorMaterial(
                    color=Color(rgba=RGBA_blue.get_with_temp_alpha(100))
                )
            ),
            fill=True,
        ),
    )  # we could use slicePartitions=3 (the minimum), but this makes the fill wrong
    return (sensor2,)


@app.cell
def _(
    Color,
    Ellipsoid,
    EllipsoidRadii,
    Material,
    RGBA_blue,
    RGBA_white,
    SolidColorMaterial,
    np,
    packets,
):
    sensor3 = packets.sensor(
        np.array([[31.4], [35.12], [1000]]),
        10,
        30,
        300,
        20,
        10_000,
        5000,
        name="A Confusing Ellipsoid Sensor",  # kwargs add to the CZML3 packet
        ellipsoid=Ellipsoid(
            radii=EllipsoidRadii(
                cartesian=[0, 0, 0]
            ),  # required to create an Ellipsoid(), and is ignored by czml3-ext
            material=Material(
                solidColor=SolidColorMaterial(
                    color=Color(rgba=RGBA_blue.get_with_temp_alpha(100))
                )
            ),
            outlineColor=Color(rgba=RGBA_white),
            fill=True,
            outline=True,
        ),
        max_ellipsoid_angle=360,
    )
    return (sensor3,)


@app.cell
def _(
    Color,
    Ellipsoid,
    EllipsoidRadii,
    Material,
    RGBA_blue,
    RGBA_white,
    SolidColorMaterial,
    np,
    packets,
):
    sensor4 = packets.sensor(
        np.array([[31.4], [35.34], [1000]]),
        10,
        30,
        300,
        20,
        10_000,
        5000,
        name="A Bad Ellipsoid Sensor",  # kwargs add to the CZML3 packet
        ellipsoid=Ellipsoid(
            radii=EllipsoidRadii(
                cartesian=[0, 0, 0]
            ),  # required to create an Ellipsoid(), and is ignored by czml3-ext
            material=Material(
                solidColor=SolidColorMaterial(
                    color=Color(rgba=RGBA_blue.get_with_temp_alpha(100))
                )
            ),
            outlineColor=Color(rgba=RGBA_white),
            fill=True,
            outline=True,
            slicePartitions=3,
        ),
        max_ellipsoid_angle=360,
    )
    return (sensor4,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Multiple sensors""")
    return


@app.cell
def _(
    Color,
    Ellipsoid,
    EllipsoidRadii,
    Material,
    Polyline,
    PolylineMaterial,
    PositionList,
    RGBA_blue,
    RGBA_orange,
    SolidColorMaterial,
    np,
    packets,
):
    ddm_LLA = np.array([[[31.75], [34.72], [0]], [[31.8], [34.68], [0]]])
    sensors1 = packets.sensor(
        ddm_LLA,
        [50, 90],
        [60, 30],
        [10, 50],
        [20, 20],
        [10_000, 20_000],
        [5_000, 2_000],
        name="My sensor",  # a kwarg sequence assigns each property to their appropriate sensor
        polyline=Polyline(
            positions=PositionList(
                cartesian=[0, 0, 0]
            ),  # required to create a Polyline(), and is ignored by czml3-ext
            material=PolylineMaterial(
                solidColor=SolidColorMaterial(color=Color(rgba=RGBA_orange))
            ),
        ),  # a kwarg that is not a sequence will be applied to all sensors
        ellipsoid=Ellipsoid(
            radii=EllipsoidRadii(
                cartesian=[0, 0, 0]
            ),  # required to create an Ellipsoid(), and is ignored by czml3-ext
            material=Material(
                solidColor=SolidColorMaterial(
                    color=Color(rgba=RGBA_blue.get_with_temp_alpha(100))
                )
            ),
            outlineColor=Color(rgba=RGBA_orange),
            fill=True,
            outline=True,
        ),
    )
    return ddm_LLA, sensors1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Grid
        Create a grid in Israel
        """
    )
    return


@app.cell
def _(get_border, shapely):
    ddm_LLA_israel = get_border("israel")
    poly_shape = shapely.Polygon(ddm_LLA_israel[:, [0, 1], 0].tolist())
    return ddm_LLA_israel, poly_shape


@app.cell
def _(ddm_LLA_israel, np, poly_shape, shapely):
    x = np.linspace(ddm_LLA_israel[:, 1, 0].min(), ddm_LLA_israel[:, 1, 0].max(), 20)
    y = np.linspace(ddm_LLA_israel[:, 0, 0].min(), ddm_LLA_israel[:, 0, 0].max(), 20)
    xv, yv = np.meshgrid(x, y)
    ddm_points = []
    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            if not poly_shape.contains(shapely.Point(yv[j, i], xv[j, i])):
                continue
            ddm_points.append([yv[j, i], xv[j, i], 0])
    ddm_LLA_points = np.array(ddm_points).reshape((-1, 3, 1))
    return ddm_LLA_points, ddm_points, i, j, x, xv, y, yv


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Define the colour palette""")
    return


@app.cell
def _(RGBA_blue, RGBA_orange, RGBA_white, create_palette, ddm_LLA_points):
    rgba = create_palette(
        [RGBA_blue, RGBA_white, RGBA_orange, RGBA_blue], ddm_LLA_points.shape[0]
    )
    return (rgba,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Create the grid, and cut it using the border""")
    return


@app.cell
def _(
    Color,
    Material,
    Polygon,
    PositionList,
    SolidColorMaterial,
    ddm_LLA_israel,
    ddm_LLA_points,
    packets,
    rgba,
):
    grid = packets.grid(
        ddm_LLA_points,
        polygon=[
            Polygon(
                positions=PositionList(cartographicDegrees=[0, 0, 0]),
                material=Material(solidColor=SolidColorMaterial(color=Color(rgba=c))),
                outlineColor=Color(rgba=[255, 255, 255, 255]),
                outline=True,
            )
            for c in rgba
        ],
        name=[f"Square {i}" for i in range(len(rgba))],
        description=[
            f"Colour: [{c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f}, {c[3]:.2f}]" for c in rgba
        ],
        ddm_LLA_cut=ddm_LLA_israel,
    )
    return (grid,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Export
        Export to file as you would any other `czml3.Document`.
        """
    )
    return


@app.cell
def _(
    CZML_VERSION,
    Document,
    Packet,
    border,
    borders,
    grid,
    sensor1,
    sensor2,
    sensor3,
    sensor4,
    sensors1,
):
    doc = Document(
        packets=(
            [Packet(name="example", id="document", version=CZML_VERSION)]
            + border
            + borders
            + sensor1
            + sensor2
            + sensor3
            + sensor4
            + sensors1
            + grid
        )
    )
    return (doc,)


@app.cell
def _(doc, fdir):
    with open(fdir / "example.czml", "w") as f:
        f.write(doc.dumps())
    return (f,)


if __name__ == "__main__":
    app.run()
