import marimo

__generated_with = "0.10.13"
app = marimo.App()


@app.cell
def _():
    import datetime
    import pathlib

    import marimo as mo
    import numpy as np
    import shapely
    from czml3 import CZML_VERSION, Document, Packet
    from czml3.properties import (
        Clock,
        Color,
        Material,
        Polygon,
        PositionList,
        SolidColorMaterial,
    )
    from czml3.types import IntervalValue, TimeIntervalCollection

    import czml3_ext
    from czml3_ext import packets
    from czml3_ext.colours import RGBA_blue, RGBA_orange, RGBA_white, create_palette
    from czml3_ext.helpers import get_border

    return (
        CZML_VERSION,
        Clock,
        Color,
        Document,
        IntervalValue,
        Material,
        Packet,
        Polygon,
        PositionList,
        RGBA_blue,
        RGBA_orange,
        RGBA_white,
        SolidColorMaterial,
        TimeIntervalCollection,
        create_palette,
        czml3_ext,
        datetime,
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
        ## Time
        czml3-ext works fully with czml3.
        """
    )
    return


@app.cell
def _(packets):
    border = packets.border("Israel")
    return (border,)


@app.cell
def _(get_border, np, shapely):
    ddm_LLA_israel = get_border("israel")
    poly_shape = shapely.Polygon(ddm_LLA_israel[:, [0, 1], 0].tolist())
    x = np.linspace(ddm_LLA_israel[:, 1, 0].min(), ddm_LLA_israel[:, 1, 0].max(), 20)
    y = np.linspace(ddm_LLA_israel[:, 0, 0].min(), ddm_LLA_israel[:, 0, 0].max(), 20)
    xv, yv = np.meshgrid(x, y)
    ddm_points = []
    for _i in range(x.shape[0]):
        for j in range(y.shape[0]):
            if not poly_shape.contains(shapely.Point(yv[j, _i], xv[j, _i])):
                continue
            ddm_points.append([yv[j, _i], xv[j, _i], 0])
    ddm_LLA_points = np.array(ddm_points).reshape((-1, 3, 1))
    return (
        ddm_LLA_israel,
        ddm_LLA_points,
        ddm_points,
        j,
        poly_shape,
        x,
        xv,
        y,
        yv,
    )


@app.cell
def _(RGBA_blue, RGBA_orange, RGBA_white, create_palette, ddm_LLA_points):
    rgba = create_palette(
        [RGBA_blue, RGBA_white, RGBA_orange, RGBA_blue], ddm_LLA_points.shape[0]
    )
    return (rgba,)


@app.cell
def _(Color, IntervalValue, datetime, rgba):
    start = datetime.datetime.now()
    s_step = 20
    cc = []
    for _i in range(len(rgba)):
        d = []
        for ii, c in enumerate(rgba[_i:] + rgba[:_i]):
            d.append(
                IntervalValue(
                    start=start + datetime.timedelta(seconds=s_step) * ii,
                    end=start + datetime.timedelta(seconds=s_step) * (ii + 1),
                    value=Color(rgba=c),
                )
            )
        cc.append(d)
    return c, cc, d, ii, s_step, start


@app.cell
def _(
    Color,
    Material,
    Polygon,
    PositionList,
    SolidColorMaterial,
    TimeIntervalCollection,
    cc,
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
                material=Material(
                    solidColor=SolidColorMaterial(
                        color=TimeIntervalCollection(values=c)
                    )
                ),
                outlineColor=Color(rgba=[255, 255, 255, 255]),
                outline=True,
            )
            for c in cc
        ],
        name=[f"Square {i}" for i in range(len(rgba))],
        description=[
            f"Colour: [{c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f}, {c[3]:.2f}]" for c in rgba
        ],
        ddm_LLA_cut=ddm_LLA_israel,
    )
    return (grid,)


@app.cell
def _(
    CZML_VERSION,
    Clock,
    Document,
    IntervalValue,
    Packet,
    border,
    datetime,
    grid,
    ii,
    s_step,
    start,
):
    doc = Document(
        packets=(
            [
                Packet(
                    name="simple",
                    id="document",
                    version=CZML_VERSION,
                    clock=IntervalValue(
                        start=start,
                        end=start + datetime.timedelta(seconds=s_step) * (ii + 1),
                        value=Clock(currentTime=start, multiplier=1200),
                    ),
                )
            ]
            + border
            + grid
        )
    )
    return (doc,)


@app.cell
def _(doc, fdir):
    with open(fdir / "example_time.czml", "w") as f:
        f.write(doc.dumps())
    return (f,)


if __name__ == "__main__":
    app.run()
