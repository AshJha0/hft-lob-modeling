"""Streaming reader for UDP payloads in pcap and pcapng captures, gzipped or not.

**Verification status: checked against byte fixtures built by hand from the
public format descriptions, and run once on one real capture, a gzipped pcapng
IEX HIST file read to the end without error.**  Classic pcap and the other link
types have not met a real file.

Formats: classic pcap (microsecond and nanosecond magics, either byte order) and
pcapng (section header, interface description, enhanced and simple packet
blocks; other block types are skipped, and ``if_tsresol`` is honoured).  Link
types: Ethernet with any number of VLAN tags, raw IP, Linux cooked and BSD
loopback.  IPv4, and IPv6 without extension headers.

Reads are buffered in large chunks, so a multi-gigabyte ``.gz`` streams in
constant memory and the per-packet cost is not a read call.  Enhanced packet
blocks and plain Ethernet/IPv4/UDP frames have a fast path; anything else, and
anything odd about those, goes through the general code, which reports it.

Every anomaly increments ``stats[name]``; with ``strict=True`` it also raises
:class:`PcapError`.  Names: ``truncated`` (the file ends inside a record, or the
gzip stream is cut), ``bad_block``, ``snapped`` (a capture shorter than the
datagram it holds).  Traffic that is simply not wanted is only counted:
``not_udp``, ``ip_fragment``, ``unknown_linktype``.
"""

from __future__ import annotations

import gzip
import struct
import zlib
from collections import Counter
from contextlib import contextmanager
from typing import BinaryIO, Dict, Iterator, List, NamedTuple, Optional, Tuple

_GZIP_MAGIC = b"\x1f\x8b"
_PCAPNG_SHB = 0x0A0D0D0A
_BOM = 0x1A2B3C4D
_BOM_SWAPPED = 0x4D3C2B1A
#: classic magic as read little-endian -> (byte order, ticks per second)
_PCAP_MAGICS = {
    0xA1B2C3D4: ("<", 10**6), 0xD4C3B2A1: (">", 10**6),
    0xA1B23C4D: ("<", 10**9), 0x4D3CB2A1: (">", 10**9),
}  # fmt: skip
_ETHERNET, _NULL, _RAW, _RAW_V4, _RAW_V6, _SLL = 1, 0, 101, 228, 229, 113
_VLAN_TYPES = (0x8100, 0x88A8, 0x9100)
_IPV4, _IPV6, _UDP = 0x0800, 0x86DD, 17
_CHUNK = 1 << 24  # decompressed bytes per refill
_RAW_CHUNK = 1 << 22  # compressed bytes per file read


class PcapError(ValueError):
    """A malformed capture under ``strict=True``."""


class UdpDatagram(NamedTuple):
    """One UDP payload; ``ts_ns`` is the capture time, nanoseconds since the epoch."""

    ts_ns: int
    dst_port: int
    payload: bytes


class _GzipStream:
    """Gzip members decompressed in large steps, the CRC checked by zlib.

    :meth:`read` returns whatever one step produced, so everything before a cut
    is delivered first; the next call raises :class:`EOFError`.  :mod:`gzip`
    would lose the bytes of a large read that meets the cut.
    """

    def __init__(self, raw: BinaryIO) -> None:
        self._raw = raw
        self._z = zlib.decompressobj(31)
        self._started = False

    def read(self, _size: int = -1) -> bytes:
        """The next decompressed bytes; b'' only at a clean end of file."""
        while True:
            if self._z.eof:
                rest = self._z.unused_data.lstrip(b"\0")  # zero padding, as gzip allows
                self._z = zlib.decompressobj(31)
                self._started = False
            else:
                rest = b""
            data = rest or self._raw.read(_RAW_CHUNK)
            if not data:
                if self._started:
                    raise EOFError("compressed file ended before the end of the stream")
                return b""
            self._started = True
            out = self._z.decompress(data)
            if out:
                return out


@contextmanager
def open_capture(path: str) -> Iterator[BinaryIO]:
    """Open ``path`` for reading, decompressing if its first bytes say gzip."""
    with open(path, "rb") as raw:
        gzipped = raw.read(2) == _GZIP_MAGIC
        raw.seek(0)
        yield _GzipStream(raw) if gzipped else raw  # type: ignore[misc]


def _ip_payload(data: bytes, offset: int) -> Tuple[str, int, int]:
    """Locate the UDP header: returns (status, UDP offset, end of IP payload)."""
    if offset >= len(data):
        return "snapped", 0, 0
    version = data[offset] >> 4
    if version == 4:
        if len(data) < offset + 20:
            return "snapped", 0, 0
        ihl = (data[offset] & 0x0F) * 4
        total = int.from_bytes(data[offset + 2 : offset + 4], "big")
        frag = int.from_bytes(data[offset + 6 : offset + 8], "big")
        if frag & 0x3FFF:  # more-fragments flag or a non-zero offset
            return "ip_fragment", 0, 0
        if data[offset + 9] != _UDP:
            return "not_udp", 0, 0
        return "ok", offset + ihl, offset + total
    if version == 6:
        if len(data) < offset + 40:
            return "snapped", 0, 0
        if data[offset + 6] != _UDP:
            return "not_udp", 0, 0
        length = int.from_bytes(data[offset + 4 : offset + 6], "big")
        return "ok", offset + 40, offset + 40 + length
    return "not_udp", 0, 0


def _network_offset(data: bytes, linktype: int) -> Optional[int]:
    """Offset of the IP header; None if the frame has no IP; -1 for an unknown link."""
    if linktype == _ETHERNET:
        offset, etype_at = 14, 12
    elif linktype == _SLL:
        offset, etype_at = 16, 14
    elif linktype in (_RAW, _RAW_V4, _RAW_V6):
        return 0
    elif linktype == _NULL:
        return 4
    else:
        return -1
    if len(data) < offset:
        return None
    etype = int.from_bytes(data[etype_at : etype_at + 2], "big")
    while etype in _VLAN_TYPES and len(data) >= offset + 4:
        etype = int.from_bytes(data[offset + 2 : offset + 4], "big")
        offset += 4
    return offset if etype in (_IPV4, _IPV6) else None


class CaptureReader:
    """Iterate the UDP datagrams of an open capture stream.

    Args:
        fh: binary stream with blocking ``read``, for example from
            :func:`open_capture`.
        strict: raise on an anomaly instead of only counting it.

    Attributes:
        stats: :class:`collections.Counter` of ``packets``, ``datagrams`` and
            the names in the module docstring.
    """

    def __init__(self, fh: BinaryIO, strict: bool = True) -> None:
        self.fh = fh
        self.strict = bool(strict)
        self.stats: Counter = Counter()
        self._buf = b""
        self._pos = 0

    def _anomaly(self, name: str, detail: str) -> None:
        self.stats[name] += 1
        if self.strict:
            raise PcapError(f"{name}: {detail}")

    def _fill(self, n: int) -> bool:
        """Make ``n`` bytes available if the file has them; False once cut short."""
        chunks = [self._buf[self._pos :]]
        have = len(chunks[0])
        self._pos = 0
        while have < n:
            try:
                chunk = self.fh.read(max(n, _CHUNK))
            except (EOFError, zlib.error, gzip.BadGzipFile) as exc:
                self._buf = b""
                self._anomaly("truncated", f"compressed stream: {exc}")
                return False
            if not chunk:
                break
            chunks.append(chunk)
            have += len(chunk)
        self._buf = b"".join(chunks)
        return True

    def _read(self, n: int) -> Optional[bytes]:
        """``n`` bytes, or b'' at end of file, or None once cut short (counted)."""
        if self._pos + n > len(self._buf) and not self._fill(n):
            return None
        data = self._buf[self._pos : self._pos + n]
        self._pos += len(data)
        if data and len(data) < n:
            self._anomaly("truncated", f"{len(data)} of {n} bytes")
            return None
        return data

    def _need(self, n: int) -> Optional[bytes]:
        """As :meth:`_read`, for a read inside a record, where end of file is a fault."""
        data = self._read(n)
        if data == b"" and n:
            self._anomaly("truncated", "file ends inside a record")
            return None
        return data

    def __iter__(self) -> Iterator[UdpDatagram]:
        head = self._read(4)
        if not head:
            return
        (magic,) = struct.unpack("<I", head)
        if magic == _PCAPNG_SHB:
            yield from self._pcapng(head)
        elif magic in _PCAP_MAGICS:
            yield from self._pcap(*_PCAP_MAGICS[magic])
        else:
            self._anomaly("bad_block", f"unknown capture magic {magic:#010x}")

    def _datagram(self, ts_ns: int, data: bytes, linktype: int, wire_len: int) -> Optional[UdpDatagram]:
        stats = self.stats
        stats["packets"] += 1
        if (
            linktype == _ETHERNET and len(data) >= 42 and data[12] == 8 and data[13] == 0
            and data[14] == 0x45 and data[23] == _UDP and not data[20] & 0x3F and not data[21]
        ):  # fmt: skip
            length = (data[38] << 8) | data[39]
            stop = 34 + length
            if length >= 8 and stop <= len(data) and stop <= 14 + ((data[16] << 8) | data[17]):
                stats["datagrams"] += 1
                return UdpDatagram(ts_ns, (data[36] << 8) | data[37], data[42:stop])
        offset = _network_offset(data, linktype)
        if offset == -1:
            stats["unknown_linktype"] += 1
            return None
        if offset is None:
            stats["not_udp"] += 1
            return None
        status, udp, end = _ip_payload(data, offset)
        if status == "snapped" or (status == "ok" and len(data) < udp + 8):
            self._anomaly("snapped", f"{len(data)} of {wire_len} bytes captured")
            return None
        if status != "ok":
            stats[status] += 1
            return None
        dst_port, length = struct.unpack_from(">HH", data, udp + 2)
        # the UDP length, not the frame, bounds the payload: Ethernet pads short frames
        stop = udp + length
        if length < 8 or stop > len(data) or stop > end:
            self._anomaly("snapped", f"UDP length {length} overruns the {len(data)}-byte frame")
            return None
        stats["datagrams"] += 1
        return UdpDatagram(ts_ns, dst_port, data[udp + 8 : stop])

    def _pcap(self, order: str, per_second: int) -> Iterator[UdpDatagram]:
        rest = self._need(20)
        if rest is None:
            return
        linktype = struct.unpack(order + "I", rest[16:20])[0] & 0x0FFFFFFF
        scale = 10**9 // per_second
        record = struct.Struct(order + "IIII")
        while True:
            head = self._read(16)
            if not head:
                return
            sec, frac, caplen, wirelen = record.unpack(head)
            data = self._need(caplen)
            if data is None:
                return
            item = self._datagram(sec * 10**9 + frac * scale, data, linktype, wirelen)
            if item is not None:
                yield item

    def _pcapng(self, first: bytes) -> Iterator[UdpDatagram]:
        order = "<"
        interfaces: List[Tuple[int, int]] = []  # (linktype, ticks per second)
        head: Optional[bytes] = first
        while True:
            if not head:
                return
            is_section = struct.unpack("<I", head)[0] == _PCAPNG_SHB  # a palindrome
            lead = self._need(8 if is_section else 4)
            if lead is None:
                return
            if is_section:
                bom = struct.unpack("<I", lead[4:8])[0]
                if bom not in (_BOM, _BOM_SWAPPED):
                    self._anomaly("bad_block", f"byte-order magic {bom:#010x}")
                    return
                order = "<" if bom == _BOM else ">"
                interfaces = []  # interface ids are scoped to a section
            btype = struct.unpack(order + "I", head)[0]
            total = struct.unpack(order + "I", lead[:4])[0]
            consumed = 4 + len(lead)
            if total % 4 or total < consumed + 4:
                self._anomaly("bad_block", f"block length {total}")
                return
            body = self._need(total - consumed)
            if body is None:
                return
            if struct.unpack(order + "I", body[-4:])[0] != total:
                self._anomaly("bad_block", "trailing length differs from leading length")
                return
            body = body[:-4]
            if btype == 1:
                interfaces.append(self._interface(body, order))
            elif btype == 6:
                yield from self._enhanced(body, order, interfaces)
            elif btype == 3:
                if not interfaces or len(body) < 4:
                    self._anomaly("bad_block", "simple packet without an interface")
                else:
                    wire = struct.unpack_from(order + "I", body)[0]
                    item = self._datagram(0, body[4 : 4 + wire], interfaces[0][0], wire)
                    if item is not None:
                        yield item
            yield from self._enhanced_run(order, interfaces)
            head = self._read(4)

    def _enhanced_run(self, order: str, interfaces: List[Tuple[int, int]]) -> Iterator[UdpDatagram]:
        """Sound enhanced packet blocks already buffered; stops at anything else."""
        lead = struct.Struct(order + "IIIIIII")
        tail = struct.Struct(order + "I")
        buf, pos = self._buf, self._pos
        size = len(buf)
        count = len(interfaces)
        while pos + 28 <= size:
            btype, total, iface, high, low, caplen, wirelen = lead.unpack_from(buf, pos)
            end = pos + total
            if (
                btype != 6 or total & 3 or total < 32 or end > size or iface >= count
                or 32 + caplen > total or tail.unpack_from(buf, end - 4)[0] != total
            ):  # fmt: skip
                break
            linktype, per_second = interfaces[iface]
            ts_ns = ((high << 32) | low) * 10**9 // per_second
            item = self._datagram(ts_ns, buf[pos + 28 : pos + 28 + caplen], linktype, wirelen)
            pos = self._pos = end
            if item is not None:
                yield item

    @staticmethod
    def _interface(body: bytes, order: str) -> Tuple[int, int]:
        linktype = struct.unpack_from(order + "H", body)[0]
        options: Dict[int, bytes] = {}
        i = 8
        while i + 4 <= len(body):
            code, length = struct.unpack_from(order + "HH", body, i)
            if code == 0:
                break
            options[code] = body[i + 4 : i + 4 + length]
            i += 4 + length + (-length % 4)
        resol = options.get(9)
        if not resol:
            return linktype, 10**6
        base = 2 if resol[0] & 0x80 else 10
        return linktype, base ** (resol[0] & 0x7F)

    def _enhanced(self, body: bytes, order: str, interfaces: List[Tuple[int, int]]) -> Iterator[UdpDatagram]:
        if len(body) < 20:
            self._anomaly("bad_block", "short enhanced packet block")
            return
        iface, high, low, caplen, wirelen = struct.unpack_from(order + "IIIII", body)
        if iface >= len(interfaces) or 20 + caplen > len(body):
            self._anomaly("bad_block", f"interface {iface}, captured length {caplen}")
            return
        linktype, per_second = interfaces[iface]
        ts_ns = ((high << 32) | low) * 10**9 // per_second
        item = self._datagram(ts_ns, body[20 : 20 + caplen], linktype, wirelen)
        if item is not None:
            yield item
