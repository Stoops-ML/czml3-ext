from pathlib import Path

"""
This folder contains:
    - A collection of border files in longitude, latitude format.
    - A collection of base64 encoded billboards that can be used directly as a billboard.
"""

BORDER_SUFFIX = ".border"  # csv file (comma delimitered) of longitude, latitude
BILLBOARD_SUFFIX = ".billboard"  # plain text


available_borders = [
    f.name for f in Path(__file__).parent.iterdir() if f.suffix == BORDER_SUFFIX
]
available_billboards = [
    f.name for f in Path(__file__).parent.iterdir() if f.suffix == BILLBOARD_SUFFIX
]
