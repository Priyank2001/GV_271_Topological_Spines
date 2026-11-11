"""
Reading and writing the VTK XML files AnyEG uses, without the VTK library.

AnyEG takes a ``.vti`` (ImageData) file and writes its extremum graph as a
``.vtp`` (PolyData) file of line segments with a ``NodeType`` point array.
VTK itself is not installed here (and is large), so this module implements
the small part of the VTK XML format those files need:

* ``read_vti(path)``    -> ``ScalarField``  (ascii, inline binary or appended
                           data; base64 or raw; zlib-compressed or not)
* ``write_vti(path, field)``                (inline base64, zlib)
* ``read_vtp(path)``    -> ``PolyLines``    (the same encodings)
* ``write_vtp(path, polylines)``            (inline base64, zlib)

Files written here open in ParaView and in the original AnyEG.

Format notes (from the VTK file-format documentation): a binary data block
starts with a header of 32- or 64-bit words (``header_type``). Uncompressed:
one word, the number of bytes. Compressed: ``[nblocks, block size, last block
size, compressed size of each block]`` followed by the zlib blocks. In base64,
VTK encodes the header and the data as separate strings, each with its own
``=`` padding; decoding every padded piece in turn handles both that and a
single joint string.
"""

from __future__ import annotations

import base64
import re
import xml.etree.ElementTree as ET
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from .field import ScalarField

PathLike = Union[str, Path]

_VTK_TYPES = {
    "Int8": "i1", "UInt8": "u1", "Int16": "i2", "UInt16": "u2", "Int32": "i4", "UInt32": "u4",
    "Int64": "i8", "UInt64": "u8", "Float32": "f4", "Float64": "f8",
}
_NUMPY_TO_VTK = {np.dtype(v).newbyteorder("<"): k for k, v in _VTK_TYPES.items()}


# ---------------------------------------------------------------------------
# Poly lines (the output of AnyEG)
# ---------------------------------------------------------------------------
@dataclass
class PolyLines:
    """Points plus line cells, as in a VTK PolyData with only ``Lines``."""

    points: np.ndarray                    # (P, 3) float
    connectivity: np.ndarray              # point ids of all lines, one after the other
    offsets: np.ndarray                   # end of each line in ``connectivity``
    point_data: Dict[str, np.ndarray] = field(default_factory=dict)

    @classmethod
    def from_segments(cls, points: np.ndarray, segments: np.ndarray, **point_data) -> "PolyLines":
        segments = np.asarray(segments, dtype=np.int64).reshape(-1, 2)
        return cls(np.asarray(points, dtype=np.float64), segments.reshape(-1),
                   2 * np.arange(1, segments.shape[0] + 1, dtype=np.int64), dict(point_data))

    @property
    def num_lines(self) -> int:
        return self.offsets.shape[0]

    def segments(self) -> np.ndarray:
        """(S, 2) every pair of consecutive points of every line."""
        starts = np.r_[0, self.offsets[:-1]]
        pairs = [np.stack([self.connectivity[a:b - 1], self.connectivity[a + 1:b]], axis=1)
                 for a, b in zip(starts, self.offsets) if b - a >= 2]
        return np.concatenate(pairs) if pairs else np.empty((0, 2), dtype=np.int64)


# ---------------------------------------------------------------------------
# Decoding data arrays
# ---------------------------------------------------------------------------
class _VtkXmlFile:
    """The parsed XML tree of a VTK XML file plus its appended-data payload."""

    def __init__(self, path: PathLike):
        raw = Path(path).read_bytes()
        cut = raw.find(b"<AppendedData")
        if cut < 0:
            self.root = ET.fromstring(raw)
            self.payload, self.payload_is_raw = b"", False
            return
        # Parse only the part before the appended data; close the root by hand.
        self.root = ET.fromstring(raw[:cut] + b"</VTKFile>")
        tag_end = raw.index(b">", cut)
        encoding = re.search(rb'encoding="(\w+)"', raw[cut:tag_end])
        self.payload_is_raw = bool(encoding) and encoding.group(1) == b"raw"
        start = raw.index(b"_", tag_end) + 1
        end = raw.rfind(b"</AppendedData>")
        self.payload = raw[start:end]
        if not self.payload_is_raw:
            self.payload = re.sub(rb"\s+", b"", self.payload)

    @property
    def header_dtype(self) -> np.dtype:
        order = "<" if self.root.get("byte_order", "LittleEndian") == "LittleEndian" else ">"
        return np.dtype(order + ("u8" if self.root.get("header_type") == "UInt64" else "u4"))

    @property
    def compressor(self) -> Optional[str]:
        return self.root.get("compressor")

    def appended_offsets(self) -> List[int]:
        return sorted(int(a.get("offset")) for a in self.root.iter("DataArray")
                      if a.get("format") == "appended")

    def read_array(self, element: ET.Element) -> np.ndarray:
        """Decode one ``<DataArray>`` element into a flat NumPy array."""
        order = "<" if self.root.get("byte_order", "LittleEndian") == "LittleEndian" else ">"
        dtype = np.dtype(order + _VTK_TYPES[element.get("type")])
        fmt = element.get("format", "ascii")
        if fmt == "ascii":
            return np.array((element.text or "").split(), dtype=dtype)
        if fmt == "binary":
            blob = _b64decode_pieces(re.sub(r"\s+", "", element.text or "").encode())
        elif fmt == "appended":
            offset = int(element.get("offset"))
            if self.payload_is_raw:
                blob = self.payload[offset:]
            else:
                later = [o for o in self.appended_offsets() if o > offset]
                end = later[0] if later else len(self.payload)
                blob = _b64decode_pieces(self.payload[offset:end])
        else:
            raise ValueError(f"unknown DataArray format {fmt!r}")
        return np.frombuffer(self._unpack(blob), dtype=dtype).copy()

    def _unpack(self, blob: bytes) -> bytes:
        """Strip the size header and decompress if needed."""
        h = self.header_dtype
        if self.compressor is None:
            nbytes = int(np.frombuffer(blob[:h.itemsize], dtype=h)[0])
            return blob[h.itemsize:h.itemsize + nbytes]
        if "ZLib" not in self.compressor:
            raise NotImplementedError(f"compressor {self.compressor} is not supported (only zlib)")
        nblocks = int(np.frombuffer(blob[:h.itemsize], dtype=h)[0])
        header = np.frombuffer(blob[:(3 + nblocks) * h.itemsize], dtype=h).astype(np.int64)
        pos = (3 + nblocks) * h.itemsize
        out = []
        for size in header[3:]:
            out.append(zlib.decompress(blob[pos:pos + size]))
            pos += size
        return b"".join(out)


def _b64decode_pieces(text: bytes) -> bytes:
    """Decode base64 made of one or more separately padded strings."""
    return b"".join(base64.b64decode(p) for p in re.findall(rb"[A-Za-z0-9+/]+={0,2}", text))


def _encode_array(values: np.ndarray, compress: bool) -> str:
    """Inline base64 text of one array (64-bit header), optionally one zlib block."""
    data = np.ascontiguousarray(values).astype(values.dtype.newbyteorder("<"), copy=False).tobytes()
    if compress:
        body = zlib.compress(data)
        header = np.array([1, len(data), len(data), len(body)], dtype="<u8").tobytes()
    else:
        body = data
        header = np.array([len(data)], dtype="<u8").tobytes()
    return (base64.b64encode(header) + base64.b64encode(body)).decode()


def _data_array_xml(name: str, values: np.ndarray, components: int, compress: bool) -> str:
    values = np.asarray(values)
    vtk_type = _NUMPY_TO_VTK[values.dtype.newbyteorder("<")]
    return (f'<DataArray type="{vtk_type}" Name="{name}" NumberOfComponents="{components}" '
            f'format="binary">{_encode_array(values.reshape(-1), compress)}</DataArray>')


def _file_header(kind: str, compress: bool) -> str:
    comp = ' compressor="vtkZLibDataCompressor"' if compress else ""
    return (f'<?xml version="1.0"?>\n<VTKFile type="{kind}" version="1.0" '
            f'byte_order="LittleEndian" header_type="UInt64"{comp}>\n')


# ---------------------------------------------------------------------------
# ImageData (.vti)
# ---------------------------------------------------------------------------
def read_vti(path: PathLike, array: Optional[str] = None) -> ScalarField:
    """Read one point-data array of a ``.vti`` file.

    ``array`` defaults to the array marked ``Scalars`` (else the first one),
    the same default AnyEG's ParaView filter shows. Other point arrays are put
    in ``field.extra``.
    """
    f = _VtkXmlFile(path)
    image = f.root.find("ImageData")
    if image is None:
        raise ValueError(f"{path} is not an ImageData file")
    ext = list(map(int, image.get("WholeExtent").split()))
    dims = (ext[1] - ext[0] + 1, ext[3] - ext[2] + 1, ext[5] - ext[4] + 1)
    spacing = tuple(map(float, image.get("Spacing", "1 1 1").split()))
    origin = tuple(map(float, image.get("Origin", "0 0 0").split()))

    pieces = image.findall("Piece")
    if len(pieces) != 1:
        raise NotImplementedError("only single-piece ImageData is supported")
    point_data = pieces[0].find("PointData")
    elements = point_data.findall("DataArray") if point_data is not None else []
    if not elements:
        raise ValueError(f"{path} has no point data")
    name = array or point_data.get("Scalars") or elements[0].get("Name")
    arrays = {e.get("Name"): e for e in elements}
    if name not in arrays:
        raise KeyError(f"no point array {name!r} in {path}; found {list(arrays)}")

    extra = {k: f.read_array(e) for k, e in arrays.items() if k != name}
    values = f.read_array(arrays[name])
    if int(arrays[name].get("NumberOfComponents", 1)) != 1:
        raise ValueError(f"array {name!r} has more than one component; AnyEG needs a scalar")
    return ScalarField(values, name=Path(path).stem, array_name=name, dims=dims,
                       spacing=spacing, origin=origin, extra=extra)


def write_vti(path: PathLike, field: ScalarField, compress: bool = True) -> None:
    """Write a grid field as a ``.vti`` file (readable by ParaView and AnyEG)."""
    if not field.is_grid:
        raise ValueError("write_vti needs a grid field")
    nx, ny, nz = field.dims
    extent = f"0 {nx - 1} 0 {ny - 1} 0 {nz - 1}"
    text = (_file_header("ImageData", compress)
            + f'<ImageData WholeExtent="{extent}" Origin="{" ".join(map(str, field.origin))}" '
            f'Spacing="{" ".join(map(str, field.spacing))}">\n<Piece Extent="{extent}">\n'
            f'<PointData Scalars="{field.array_name}">\n'
            + _data_array_xml(field.array_name, field.values, 1, compress)
            + "\n</PointData>\n<CellData/>\n</Piece>\n</ImageData>\n</VTKFile>\n")
    Path(path).write_text(text)


# ---------------------------------------------------------------------------
# PolyData (.vtp)
# ---------------------------------------------------------------------------
def read_vtp(path: PathLike) -> PolyLines:
    """Read the points, lines and point arrays of a ``.vtp`` file."""
    f = _VtkXmlFile(path)
    piece = f.root.find("PolyData/Piece")
    if piece is None:
        raise ValueError(f"{path} is not a PolyData file")
    points = f.read_array(piece.find("Points/DataArray")).astype(np.float64).reshape(-1, 3)
    lines = {e.get("Name"): e for e in piece.findall("Lines/DataArray")}
    if lines:
        conn = f.read_array(lines["connectivity"]).astype(np.int64)
        offsets = f.read_array(lines["offsets"]).astype(np.int64)
    else:
        conn = offsets = np.empty(0, dtype=np.int64)
    pd = {e.get("Name"): f.read_array(e) for e in piece.findall("PointData/DataArray")}
    return PolyLines(points, conn, offsets, pd)


def write_vtp(path: PathLike, lines: PolyLines, compress: bool = True) -> None:
    """Write ``PolyLines`` as a ``.vtp`` file (point arrays become PointData)."""
    n_pts = lines.points.shape[0]
    pd = "".join(_data_array_xml(name, np.asarray(a), 1 if np.ndim(a) == 1 else np.shape(a)[1],
                                 compress) + "\n" for name, a in lines.point_data.items())
    text = (_file_header("PolyData", compress)
            + f'<PolyData>\n<Piece NumberOfPoints="{n_pts}" NumberOfVerts="0" '
            f'NumberOfLines="{lines.num_lines}" NumberOfStrips="0" NumberOfPolys="0">\n'
            f"<PointData>\n{pd}</PointData>\n<CellData/>\n<Points>\n"
            + _data_array_xml("Points", lines.points.astype(np.float32), 3, compress)
            + "\n</Points>\n<Lines>\n"
            + _data_array_xml("connectivity", lines.connectivity.astype(np.int64), 1, compress) + "\n"
            + _data_array_xml("offsets", lines.offsets.astype(np.int64), 1, compress)
            + "\n</Lines>\n</Piece>\n</PolyData>\n</VTKFile>\n")
    Path(path).write_text(text)
