import json
from tempfile import mktemp

import pytest
from czml3 import CZML_VERSION, Document, Packet

from czml3_ext.data import available_billboards, available_borders
from czml3_ext.errors import BillboardNotFound, BorderNotFound
from czml3_ext.helpers import combine_docs, get_billboard, get_border


def test_billboard_names():
    for b in available_billboards:
        assert b.lower() == b


def test_border_names():
    for b in available_borders:
        assert b.lower() == b


def test_billboard_not_found():
    with pytest.raises(BillboardNotFound):
        _ = get_billboard("F21")
    with pytest.raises(BillboardNotFound):
        _ = get_billboard("F22.csv")


def test_border_not_found():
    with pytest.raises(BorderNotFound):
        _ = get_border("F21")
    with pytest.raises(BorderNotFound):
        _ = get_border("F22.csv")


def test_get_border():
    for b in available_borders:
        get_border(b)
        get_border(b.upper())
        get_border(b.lower())
        get_border(b.removeprefix(".border"))


def test_get_billboard():
    for b in available_billboards:
        get_billboard(b)
        get_billboard(b.upper())
        get_billboard(b.lower())
        get_billboard(b.removeprefix(".billboard"))


def test_combine_docs():
    p0 = [
        {
            "id": "document",
            "name": "simple",
            "version": "1.0",
            "clock": {
                "interval": "2012-03-15T10:00:00.000000Z/2012-03-16T10:00:00.000000Z",
                "currentTime": "2012-03-15T10:00:00.000000Z",
                "multiplier": 60,
            },
        },
        {
            "id": "Satellite/Molniya_1-92",
            "name": "Molniya_1-92",
            "availability": "2012-03-15T10:00:00.000000Z/2012-03-16T10:00:00.000000Z",
        },
    ]
    p1 = [
        {
            "id": "document",
            "name": "simple",
            "version": "1.0",
        },
        {
            "id": "CAR",
            "name": "CAR",
            "availability": "2012-03-15T10:00:00.000000Z/2012-03-16T12:00:00.000000Z",
        },
    ]
    p2 = [{"id": "PLANE", "name": "PLANE"}]
    f = mktemp(".czml")
    with open(f, "w") as fp:
        json.dump(
            [
                {
                    "id": "document",
                    "name": "simple",
                    "version": "1.0",
                    "clock": {
                        "interval": "2012-03-15T10:00:00.000000Z/2012-03-16T10:00:00.000000Z",
                        "currentTime": "2012-03-15T10:00:00.000000Z",
                        "multiplier": 60,
                    },
                },
                {
                    "id": "9927edc4-e87a-4e1f-9b8b-0bfb3b05b227",
                    "name": "Accesses",
                    "description": "List of Accesses",
                },
            ],
            fp,
        )
    d = Document(
        packets=[
            Packet(id="document", version=CZML_VERSION, name="test"),
            Packet(id="TRAIN", name="TRAIN"),
        ]
    )
    out = combine_docs((p0, p1, f, p2, d), 1)
    assert out == json.dumps(
        [
            {
                "id": "document",
                "name": "simple",
                "version": "1.0",
            },
            {
                "id": "Satellite/Molniya_1-92",
                "name": "Molniya_1-92",
                "availability": "2012-03-15T10:00:00.000000Z/2012-03-16T10:00:00.000000Z",
            },
            {
                "id": "CAR",
                "name": "CAR",
                "availability": "2012-03-15T10:00:00.000000Z/2012-03-16T12:00:00.000000Z",
            },
            {
                "id": "9927edc4-e87a-4e1f-9b8b-0bfb3b05b227",
                "name": "Accesses",
                "description": "List of Accesses",
            },
            {"id": "PLANE", "name": "PLANE"},
            {"id": "TRAIN", "name": "TRAIN"},
        ]
    )
