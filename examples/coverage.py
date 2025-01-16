import marimo

__generated_with = "0.10.13"
app = marimo.App()


@app.cell
def _():
    import pathlib

    import marimo as mo
    import numpy as np
    import rasterio
    from czml3 import CZML_VERSION, Document, Packet
    from czml3.properties import (
        Color,
        Material,
        Polygon,
        PositionList,
        SolidColorMaterial,
    )

    import czml3_ext
    from czml3_ext import packets, rasters
    from czml3_ext.colours import (
        RGBA_black,
        RGBA_white,
        create_palette,
    )

    return (
        CZML_VERSION,
        Color,
        Document,
        Material,
        Packet,
        Polygon,
        PositionList,
        RGBA_black,
        RGBA_white,
        SolidColorMaterial,
        create_palette,
        czml3_ext,
        mo,
        np,
        packets,
        pathlib,
        rasterio,
        rasters,
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
    mo.md(r"""## Coverage""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Coverage of a single sensor""")
    return


@app.cell
def _(fdir, packets, rasters):
    vs = packets.coverage(rasters.ops(fdir / "data.tif", 5), delete_rasters=True)
    return (vs,)


@app.cell
def _(fdir, rasterio):
    with rasterio.open(fdir / "data.tif") as src:
        arr = src.read(1)
        origin_x, origin_y = src.transform * (0, 0)
        delta_x = src.transform[0]
        delta_y = src.transform[4]
    return arr, delta_x, delta_y, origin_x, origin_y, src


@app.cell
def _(arr, delta_x, delta_y, np, origin_x, origin_y, packets):
    sensor_coverage = packets.sensor(
        np.array(
            [
                [origin_y + arr.shape[0] / 2 * delta_y],
                [origin_x + arr.shape[1] / 2 * delta_x],
                [0],
            ]
        ),
        320,
        20,
        180,
        20,
        10_000,
    )
    return (sensor_coverage,)


@app.cell
def _(CZML_VERSION, Document, Packet, fdir, sensor_coverage, vs):
    _doc = Document(
        packets=[Packet(name="example", id="document", version=CZML_VERSION)]
        + vs
        + sensor_coverage
    )
    with open(fdir / "example.czml", "w") as _f:
        _f.write(_doc.dumps())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Coverage with a hole""")
    return


@app.cell
def _(fdir, packets, rasters):
    vs_hole = packets.coverage(
        rasters.ops(fdir / "data.tif", 5),
        rasters.ops(fdir / "data_cut.tif", 5),
        delete_rasters=True,
    )
    return (vs_hole,)


@app.cell
def _(CZML_VERSION, Document, Packet, fdir, sensor_coverage, vs_hole):
    _doc = Document(
        packets=[Packet(name="example", id="document", version=CZML_VERSION)]
        + vs_hole
        + sensor_coverage
    )
    with open(fdir / "example.czml", "w") as _f:
        _f.write(_doc.dumps())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Coverage above a height
        Show areas that are above 100 metres.
        """
    )
    return


@app.cell
def _(fdir, packets, rasters):
    vs_height = packets.coverage(
        rasters.ops(fdir / "data.tif", 100, "ge", band=2), delete_rasters=True
    )
    return (vs_height,)


@app.cell
def _(CZML_VERSION, Document, Packet, fdir, sensor_coverage, vs_height):
    _doc = Document(
        packets=[Packet(name="example", id="document", version=CZML_VERSION)]
        + vs_height
        + sensor_coverage
    )
    with open(fdir / "example.czml", "w") as _f:
        _f.write(_doc.dumps())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Coverage between heights
        Show areas that are between 100 metres and 200 metres. The method below uses a hole to remove areas below 100 metres.
        """
    )
    return


@app.cell
def _(fdir, packets, rasters):
    vs_heights = packets.coverage(
        rasters.ops(fdir / "data.tif", 200, "le", band=2),
        rasters.ops(fdir / "data.tif", 100, "le", band=2),
        delete_rasters=True,
    )
    return (vs_heights,)


@app.cell
def _(CZML_VERSION, Document, Packet, fdir, sensor_coverage, vs_heights):
    _doc = Document(
        packets=[Packet(name="example", id="document", version=CZML_VERSION)]
        + vs_heights
        + sensor_coverage
    )
    with open(fdir / "example.czml", "w") as _f:
        _f.write(_doc.dumps())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""The next method chains operations on the same raster and therefore doesn't require inputting holes to `packets.coverage()`. This is the recommended approach as it is more performant."""
    )
    return


@app.cell
def _(fdir, packets, rasters):
    vs_heights_1 = packets.coverage(
        rasters.ops(fdir / "data.tif", [200, 100], ["le", "ge"], band=2),
        delete_rasters=True,
    )
    return (vs_heights_1,)


@app.cell
def _(CZML_VERSION, Document, Packet, fdir, sensor_coverage, vs_heights_1):
    _doc = Document(
        packets=[Packet(name="example", id="document", version=CZML_VERSION)]
        + vs_heights_1
        + sensor_coverage
    )
    with open(fdir / "example.czml", "w") as _f:
        _f.write(_doc.dumps())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""`rasters.ops()` will raise an error if the output raster is empty. This can be turned off using the `error_if_no_data` input."""
    )
    return


@app.cell
def _(fdir, rasters):
    try:
        rasters.ops(fdir / "data.tif", -999)  # error_if_no_data=True by default
    except ValueError as e:
        print(e)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Amount of coverage overlap
        Show polygons of the amount of overlap. The amount of overlap corresponds to a colour on a predefined scale.
        """
    )
    return


@app.cell
def _(fdir, rasters):
    fpaths = [fdir / "data_cut.tif", fdir / "data.tif"]
    fpath_coverage, num_coverages = rasters.coverage_amount(fpaths, 5)
    return fpath_coverage, fpaths, num_coverages


@app.cell
def _(RGBA_black, RGBA_white, create_palette, fpaths):
    rgba = create_palette([RGBA_black, RGBA_white], len(fpaths) + 1)
    return (rgba,)


@app.cell
def _(
    Color,
    Material,
    Polygon,
    PositionList,
    SolidColorMaterial,
    fpath_coverage,
    num_coverages,
    packets,
    rasters,
    rgba,
):
    overlaps = []
    for num_covers in num_coverages:
        overlaps.extend(
            packets.coverage(
                rasters.ops(fpath_coverage, num_covers),
                name=f"Overlap of {num_covers}",
                polygon=Polygon(
                    positions=PositionList(cartographicDegrees=[0, 0, 0]),
                    material=Material(
                        solidColor=SolidColorMaterial(
                            color=Color(rgba=rgba[num_covers])
                        )
                    ),
                ),
                delete_rasters=True,
            )
        )
    return num_covers, overlaps


@app.cell
def _(CZML_VERSION, Document, Packet, fdir, overlaps):
    _doc = Document(
        packets=[Packet(name="example", id="document", version=CZML_VERSION)] + overlaps
    )
    with open(fdir / "example.czml", "w") as _f:
        _f.write(_doc.dumps())
    return


if __name__ == "__main__":
    app.run()
