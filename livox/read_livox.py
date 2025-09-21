"""Classes for reading and writing livox .lvx files. Updated to V1.3 standards in according to
Livox-SDK. Specifically adapted for Livox Avia support.

Documentation was updated through referencing the following sheets:
https://github.com/Livox-SDK/Livox-SDK/wiki/Livox-SDK-Communication-Protocol
https://terra-1-g.djicdn.com/65c028cd298f4669a7f0e40e50ba1131/Download/LVX%20Specifications.pdf

"""

from __future__ import annotations

from struct import Struct
from dataclasses import dataclass, field
from typing import Tuple, List, BinaryIO
from collections.abc import Iterable, Iterator

import os
import pandas as pd

class LvxParseError(Exception):
    """An error during reading / writing of a .lvx file."""

    pass


@dataclass
class LvxDeviceInfo:
    """A single device info block."""

    lidar_broadcast_code: bytes = b'\x00'*16
    hub_broadcast_code: bytes = b'\x00'*16
    device_index: int = 0
    device_type: int = 0
    extrinsic_enabled: bool = False
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    struct = Struct('=16s 16s B B B 6f')
    _in_pos = -1

    @property
    def size(self):
        """Get the size in bytes."""
        return self.struct.size

    def write_to(self, fp: BinaryIO):
        """Write the structure to a binary file."""
        struct = self.struct
        data = struct.pack(self.lidar_broadcast_code,
                           self.hub_broadcast_code,
                           self.device_index,
                           self.device_type,
                           self.extrinsic_enabled,
                           self.roll,
                           self.pitch,
                           self.yaw,
                           self.x,
                           self.y,
                           self.z)
        fp.write(data)

    @classmethod
    def read_from(cls, fp: BinaryIO) -> LvxDeviceInfo:
        """Read the structure from a binary file."""
        struct = cls.struct
        in_pos = fp.tell()
        data = fp.read(struct.size)
        if len(data) != struct.size:
            raise LvxParseError(
                'Not enough data, need {} got {}'.format(struct.size, len(data)))

        h = LvxDeviceInfo()
        h._in_pos = in_pos
        h.lidar_broadcast_code,\
            h.hub_broadcast_code,\
            h.device_index,\
            h.device_type,\
            h.extrinsic_enabled,\
            h.roll,\
            h.pitch,\
            h.yaw,\
            h.x,\
            h.y,\
            h.z\
            = struct.unpack(data)

        return h


@dataclass
class LvxHeader:
    """Public and private file headers."""

    signature: bytes = b'livox_tech\x00\x00\x00\x00\x00\x00'
    version: Tuple[int, int, int, int] = (1, 1, 0, 0)
    magic_code: int = 0xAC0EA767
    frame_duration: int = 50
    device_count: int = 1
    devices: List[LvxDeviceInfo] = field(default_factory=list, repr=False)

    struct = Struct('=16s 4B I I B')
    _in_pos = -1

    @property
    def size(self):
        """Get the size in bytes."""
        return self.struct.size + sum(d.size for d in self.devices)

    def write_to(self, fp: BinaryIO):
        """Write the structure to a binary file."""
        struct = self.struct

        data = struct.pack(self.signature,
                           *self.version,
                           self.magic_code,
                           self.frame_duration,
                           self.device_count)
        fp.write(data)

        for dev in self.devices:
            dev.write_to(fp)

    @classmethod
    def read_from(cls, fp: BinaryIO) -> LvxHeader:
        """Read the structure from a binary file."""
        struct = cls.struct
        in_pos = fp.tell()
        data = fp.read(struct.size)
        if len(data) != struct.size:
            raise LvxParseError(
                'Not enough data, need {} got {}'.format(struct.size, len(data)))

        h = LvxHeader()
        h._in_pos = in_pos
        h.signature,\
            v1, v2, v3, v4,\
            h.magic_code,\
            h.frame_duration,\
            h.device_count\
            = struct.unpack(data)
        h.version = (v1, v2, v3, v4)

        for _ in range(h.device_count):
            dev = LvxDeviceInfo.read_from(fp)
            h.devices.append(dev)

        return h


class LvxPoint:
    """Base class for a single point."""

    struct = Struct('')
    _in_pos = -1

    @property
    def size(self):
        """Get the size in bytes."""
        return self.struct.size

    def write_to(self, fp: BinaryIO):
        """Write the structure to a binary file."""
        raise NotImplementedError('LvxPoint.write_to is not implemented')


@dataclass
class LvxRawPoint(LvxPoint):
    """A single raw point."""

    x: int = 0  # mm
    y: int = 0  # mm
    z: int = 0  # mm
    reflectivity: int = 0

    struct = Struct('=3I B')

    def write_to(self, fp: BinaryIO):
        """Write the structure to a binary file."""
        struct = self.struct

        data = struct.pack(self.x,
                           self.y,
                           self.z,
                           self.reflectivity)
        fp.write(data)

    @classmethod
    def read_from(cls, fp: BinaryIO) -> LvxRawPoint:
        """Read the structure from a binary file."""
        struct = cls.struct
        in_pos = fp.tell()
        data = fp.read(struct.size)
        if len(data) != struct.size:
            raise LvxParseError(
                'Not enough data, need {} got {}'.format(struct.size, len(data)))

        p = LvxRawPoint()
        p._in_pos = in_pos
        p.x, p.y, p.z, p.reflectivity = struct.unpack(data)

        return p


@dataclass
class LvxSpherePoint(LvxPoint):
    """A single sphere point."""

    depth: int = 0  # mm
    theta: int = 0  # 0.01 deg
    phi: int = 0  # 0.01 deg
    reflectivity: int = 0

    struct = Struct('=I 2H B')

    def write_to(self, fp: BinaryIO):
        """Write the structure to a binary file."""
        struct = self.struct

        data = struct.pack(self.depth,
                           self.theta,
                           self.phi,
                           self.reflectivity)
        fp.write(data)

    @classmethod
    def read_from(cls, fp: BinaryIO) -> LvxSpherePoint:
        """Read the structure from a binary file."""
        struct = cls.struct
        in_pos = fp.tell()
        data = fp.read(struct.size)
        if len(data) != struct.size:
            raise LvxParseError(
                'Not enough data, need {} got {}'.format(struct.size, len(data)))

        p = LvxSpherePoint()
        p._in_pos = in_pos
        p.depth, p.theta, p.phi, p.reflectivity = struct.unpack(data)

        return p


@dataclass
class LvxExtendedRawPoint(LvxPoint):
    """A single extended raw point."""

    x: int = 0  # mm
    y: int = 0  # mm
    z: int = 0  # mm
    reflectivity: int = 0
    tag: int = 0

    struct = Struct('=3I 2B')

    def write_to(self, fp: BinaryIO):
        """Write the structure to a binary file."""
        struct = self.struct

        data = struct.pack(self.x,
                           self.y,
                           self.z,
                           self.reflectivity,
                           self.tag)
        fp.write(data)

    @classmethod
    def read_from(cls, fp: BinaryIO) -> LvxExtendedRawPoint:
        """Read the structure from a binary file."""
        struct = cls.struct
        in_pos = fp.tell()
        data = fp.read(struct.size)
        if len(data) != struct.size:
            raise LvxParseError(
                'Not enough data, need {} got {}'.format(struct.size, len(data)))

        p = LvxExtendedRawPoint()
        p._in_pos = in_pos
        p.x, p.y, p.z, p.reflectivity, p.tag = struct.unpack(data)

        return p


@dataclass
class LvxExtendedSpherePoint(LvxPoint):
    """A single extended sphere point."""

    depth: int = 0  # mm
    theta: int = 0  # 0.01 deg
    phi: int = 0  # 0.01 deg
    reflectivity: int = 0
    tag: int = 0

    struct = Struct('=I 2H 2B')

    def write_to(self, fp: BinaryIO):
        """Write the structure to a binary file."""
        struct = self.struct

        data = struct.pack(self.depth,
                           self.theta,
                           self.phi,
                           self.reflectivity,
                           self.tag)
        fp.write(data)

    @classmethod
    def read_from(cls, fp: BinaryIO) -> LvxExtendedSpherePoint:
        """Read the structure from a binary file."""
        struct = cls.struct
        in_pos = fp.tell()
        data = fp.read(struct.size)
        if len(data) != struct.size:
            raise LvxParseError(
                'Not enough data, need {} got {}'.format(struct.size, len(data)))

        p = LvxExtendedSpherePoint()
        p._in_pos = in_pos
        p.depth, p.theta, p.phi, p.reflectivity, p.tag = struct.unpack(data)

        return p


@dataclass
class LvxDualExtendedRawPoint(LvxPoint):
    """A single dual extended raw point."""

    x1: int = 0  # mm
    y1: int = 0  # mm
    z1: int = 0  # mm
    reflectivity1: int = 0
    tag1: int = 0
    x2: int = 0  # mm
    y2: int = 0  # mm
    z2: int = 0  # mm
    reflectivity2: int = 0
    tag2: int = 0

    struct = Struct('=3I 2B 3I 2B')

    def write_to(self, fp: BinaryIO):
        """Write the structure to a binary file."""
        struct = self.struct

        data = struct.pack(self.x1,
                           self.y1,
                           self.z1,
                           self.reflectivity1,
                           self.tag1,
                           self.x2,
                           self.y2,
                           self.z2,
                           self.reflectivity2,
                           self.tag2)
        fp.write(data)

    @classmethod
    def read_from(cls, fp: BinaryIO) -> LvxDualExtendedRawPoint:
        """Read the structure from a binary file."""
        struct = cls.struct
        in_pos = fp.tell()
        data = fp.read(struct.size)
        if len(data) != struct.size:
            raise LvxParseError(
                'Not enough data, need {} got {}'.format(struct.size, len(data)))

        p = LvxDualExtendedRawPoint()
        p._in_pos = in_pos
        p.x1, p.y1, p.z1,\
            p.reflectivity1, p.tag1,\
            p.x2, p.y2, p.z2,\
            p.reflectivity2, p.tag2,\
            = struct.unpack(data)

        return p


@dataclass
class LvxDualExtendedSpherePoint(LvxPoint):
    """A single dual extended sphere point."""

    theta: int = 0  # 0.01 deg
    phi: int = 0  # 0.01 deg
    depth1: int = 0  # mm
    reflectivity1: int = 0
    tag1: int = 0
    depth2: int = 0  # mm
    reflectivity2: int = 0
    tag2: int = 0

    struct = Struct('=2H I 2B I 2B')

    def write_to(self, fp: BinaryIO):
        """Write the structure to a binary file."""
        struct = self.struct

        data = struct.pack(self.theta,
                           self.phi,
                           self.depth1,
                           self.reflectivity1,
                           self.tag1,
                           self.depth2,
                           self.reflectivity2,
                           self.tag2)
        fp.write(data)

    @classmethod
    def read_from(cls, fp: BinaryIO) -> LvxDualExtendedSpherePoint:
        """Read the structure from a binary file."""
        struct = cls.struct
        in_pos = fp.tell()
        data = fp.read(struct.size)
        if len(data) != struct.size:
            raise LvxParseError(
                'Not enough data, need {} got {}'.format(struct.size, len(data)))

        p = LvxDualExtendedSpherePoint()
        p._in_pos = in_pos
        p.theta, p.phi,\
            p.depth1, p.reflectivity1, p.tag1,\
            p.depth2, p.reflectivity2, p.tag2,\
            = struct.unpack(data)

        return p


@dataclass
class LvxImuPoint(LvxPoint):
    """A single imu point."""

    gyro_x: float = 0.0
    gyro_y: float = 0.0
    gyro_z: float = 0.0
    acc_x: float = 0.0
    acc_y: float = 0.0
    acc_z: float = 0.0

    struct = Struct('=6f')

    def write_to(self, fp: BinaryIO):
        """Write the structure to a binary file."""
        struct = self.struct

        data = struct.pack(self.gyro_x,
                           self.gyro_y,
                           self.gyro_z,
                           self.acc_x,
                           self.acc_y,
                           self.acc_z)
        fp.write(data)

    def to_json(self):
        return {"gyro_x": self.gyro_x,
                "gyro_y": self.gyro_y,
                "gyro_z": self.gyro_z,
                "acc_x": self.acc_x,
                "acc_y": self.acc_y,
                "acc_z": self.acc_z
                 }

    @classmethod
    def read_from(cls, fp: BinaryIO) -> LvxImuPoint:
        """Read the structure from a binary file."""
        struct = cls.struct
        in_pos = fp.tell()
        data = fp.read(struct.size)
        if len(data) != struct.size:
            raise LvxParseError(
                'Not enough data, need {} got {}'.format(struct.size, len(data)))

        p = LvxImuPoint()
        p._in_pos = in_pos
        p.gyro_x, p.gyro_y, p.gyro_z,\
            p.acc_x, p.acc_y, p.acc_z,\
            = struct.unpack(data)

        return p

#TODO: Implement this class
@dataclass
class LvxTripleExtendCartesian(LvxPoint):
    """A single triple extended raw point."""

    x1: int = 0  # mm
    y1: int = 0  # mm
    z1: int = 0  # mm
    reflectivity1: int = 0
    tag1: int = 0
    x2: int = 0  # mm
    y2: int = 0  # mm
    z2: int = 0  # mm
    reflectivity2: int = 0
    tag2: int = 0
    x3: int = 0  # mm
    y3: int = 0  # mm
    z3: int = 0  # mm
    reflectivity3: int = 0
    tag3: int = 0

    struct = Struct('=3i 2B 3i 2B 3i 2B')

    def write_to(self, fp: BinaryIO):
        """Write the structure to a binary file."""
        struct = self.struct

        data = struct.pack(self.x1,
                           self.y1,
                           self.z1,
                           self.reflectivity1,
                           self.tag1,
                           self.x2,
                           self.y2,
                           self.z2,
                           self.reflectivity2,
                           self.tag2,
                           self.x3,
                           self.y3,
                           self.z3,
                           self.reflectivity3,
                           self.tag3)
        fp.write(data)

    def to_json(self):
        return {"x1": self.x1,
                "y1": self.y1,
                "z1": self.z1,
                "reflectivity1": self.reflectivity1,
                "tag1": self.tag1,
                "x2": self.x2,
                "y2": self.y2,
                "z2": self.z2,
                "reflectivity2": self.reflectivity2,
                "tag2": self.tag2,
                "x3": self.x3,
                "y3": self.y3,
                "z3": self.z3,
                "reflectivity3": self.reflectivity3,
                "tag3": self.tag3
                 }

    @classmethod
    def read_from(cls, fp: BinaryIO) -> LvxTripleExtendCartesian:
        """Read the structure from a binary file."""
        struct = cls.struct
        in_pos = fp.tell()
        data = fp.read(struct.size)
        if len(data) != struct.size:
            raise LvxParseError(
                'Not enough data, need {} got {}'.format(struct.size, len(data)))

        p = LvxTripleExtendCartesian()
        p._in_pos = in_pos
        p.x1, p.y1, p.z1,\
            p.reflectivity1, p.tag1,\
            p.x2, p.y2, p.z2,\
            p.reflectivity2, p.tag2,\
            p.x3, p.y3, p.z3,\
            p.reflectivity3, p.tag3,\
            = struct.unpack(data)

        return p
    
#TODO: Implement this class
@dataclass
class LvxTripleExtendSpherePoint(LvxPoint):
    """A single triple extended sphere point."""

    theta: int = 0  # 0.01 deg
    phi: int = 0  # 0.01 deg
    depth1: int = 0  # mm
    reflectivity1: int = 0
    tag1: int = 0
    depth2: int = 0  # mm
    reflectivity2: int = 0
    tag2: int = 0
    depth3: int = 0  # mm
    reflectivity3: int = 0
    tag3: int = 0

    struct = Struct('=2H I 2B I 2B I 2B')

    def write_to(self, fp: BinaryIO):
        """Write the structure to a binary file."""
        struct = self.struct

        data = struct.pack(self.theta,
                           self.phi,
                           self.depth1,
                           self.reflectivity1,
                           self.tag1,
                           self.depth2,
                           self.reflectivity2,
                           self.tag2,
                           self.depth3,
                           self.reflectivity3,
                           self.tag3)
        fp.write(data)

    @classmethod
    def read_from(cls, fp: BinaryIO) -> LvxTripleExtendSpherePoint:
        """Read the structure from a binary file."""
        struct = cls.struct
        in_pos = fp.tell()
        data = fp.read(struct.size)
        if len(data) != struct.size:
            raise LvxParseError(
                'Not enough data, need {} got {}'.format(struct.size, len(data)))

        p = LvxTripleExtendSpherePoint()
        p._in_pos = in_pos
        p.theta, p.phi,\
            p.depth1, p.reflectivity1, p.tag1,\
            p.depth2, p.reflectivity2, p.tag2,\
            p.depth3, p.reflectivity3, p.tag3,\
            = struct.unpack(data)

        return p

TimestampT = Tuple[int, int, int, int, int, int, int, int]

@dataclass
class LvxPackage:
    """A single package which contains some number of points."""

    device_index: int = 0
    version: int = 0
    port_id: int = 0
    lidar_index: int = 0
    # rsvd: int = 0
    error_code: int = 0
    timestamp_type: int = 0
    data_type: int = 0
    timestamp: TimestampT = (0, 0, 0, 0, 0, 0, 0, 0)
    points: List[LvxPoint] = field(default_factory=list)

    struct = Struct('=5B I B B 8s') # Changed to Match Livox SDK Github
    _in_pos = -1

    @property
    def size(self):
        """Get the size in bytes."""
        return self.struct.size + sum(p.size for p in self.points)

    def write_to(self, fp: BinaryIO):
        """Write the structure to a binary file."""
        struct = self.struct

        data = struct.pack(self.device_index,
                           self.version,
                           self.port_id,
                           self.lidar_index,
                           0,
                           self.error_code,
                           self.timestamp_type,
                           self.data_type,
                           self.timestamp)
        fp.write(data)

        for point in self.points:
            point.write_to(fp)

    @classmethod
    def read_from(cls, fp: BinaryIO, next_offset: int) -> LvxPackage:
        """Read the structure from a binary file."""
        emptyPack = False
        struct = cls.struct
        in_pos = fp.tell()
        data = fp.read(struct.size)
        if len(data) != struct.size:
            raise LvxParseError(
                'Not enough data, need {} got {}'.format(struct.size, len(data)))

        p = LvxPackage()
        p._in_pos = in_pos
        p.device_index,\
            p.version,\
            p.port_id,\
            p.lidar_index,\
            _,\
            p.error_code,\
            p.timestamp_type,\
            p.data_type,\
            p.timestamp\
            = struct.unpack(data)
        #print(p)

        if p.data_type == 0:
            count = 100
            p_type = LvxRawPoint
        elif p.data_type == 1:
            count = 100
            p_type = LvxSpherePoint
        elif p.data_type == 2:
            count = 96
            p_type = LvxExtendedRawPoint
        elif p.data_type == 3:
            count = 96
            p_type = LvxExtendedSpherePoint
        elif p.data_type == 4:
            count = 48
            p_type = LvxDualExtendedRawPoint
        elif p.data_type == 5:
            count = 48
            p_type = LvxDualExtendedSpherePoint
        elif p.data_type == 6:
            count = 1
            p_type = LvxImuPoint
        elif p.data_type == 7:
            count = 30
            p_type = LvxTripleExtendCartesian
        elif p.data_type == 8:
            count = 30
            p_type = LvxTripleExtendSpherePoint
        else:
            count = 0
            emptyPack = True
            #print("Error point Diff: ", next_offset-fp.tell())
            #raise LvxParseError('Unknown point type {}'.format(p.data_type))

        for c in range(count):
            #print(c)
            point = p_type.read_from(fp)
            p.points.append(point)

        return (p, emptyPack)


@dataclass
class LvxFrameHeader:
    """Header for a single frame."""

    current_offset: int = 0
    next_offset: int = 0
    frame_index: int = 0

    struct = Struct('=3Q')
    _in_pos = -1

    @property
    def size(self):
        """Get the size in bytes."""
        return self.struct.size

    def write_to(self, fp: BinaryIO):
        """Write the structure to a binary file."""
        struct = self.struct

        data = struct.pack(self.current_offset,
                           self.next_offset,
                           self.frame_index)
        fp.write(data)

    @classmethod
    def read_from(cls, fp: BinaryIO) -> LvxFrameHeader:
        """Read the structure from a binary file."""
        struct = cls.struct
        in_pos = fp.tell()
        data = fp.read(struct.size)
        if len(data) != struct.size:
            raise LvxParseError(
                'Not enough data, need {} got {}'.format(struct.size, len(data)))

        h = LvxFrameHeader()
        h._in_pos = in_pos
        h.current_offset,\
            h.next_offset,\
            h.frame_index\
            = struct.unpack(data)
        
        return h


@dataclass
class LvxFrame:
    """A single frame with header and packages."""

    header: LvxFrameHeader
    packages: List[LvxPackage] = field(default_factory=list, repr=False)

    _in_pos = -1

    @property
    def size(self):
        """Get the size in bytes."""
        return self.header.size + sum(p.size for p in self.packages)

    def write_to(self, fp: BinaryIO):
        """Write the structure to a binary file."""
        self.header.write_to(fp)

        for package in self.packages:
            package.write_to(fp)

    @classmethod
    def read_from(cls, fp: BinaryIO) -> LvxFrame:
        """Read the structure from a binary file."""
        in_pos = fp.tell()

        header = LvxFrameHeader.read_from(fp)
        f = LvxFrame(header)
        f._in_pos = in_pos

        while fp.tell() < f.header.next_offset:
            package, emptyPack = LvxPackage.read_from(fp, f.header.next_offset)
            if not emptyPack:
                f.packages.append(package)
            else:
                fp.seek(f.header.next_offset-fp.tell(), 1)    
                
        fp.seek(f.header.next_offset-fp.tell(), 1)  
        #print("Real Next offset: ", f.header.next_offset)
        #print("Curr Tell", fp.tell())
        assert(f.header.next_offset == fp.tell())
        return f


class LvxFrameIter(Iterator):
    """Iterator for LvxFrame's."""

    def __init__(self, fp: BinaryIO):
        """Create a new LvxFrameIter reading from fp."""
        self.fp = fp
        self.begin = fp.tell()

        fp.seek(0, 2)
        self.end = fp.tell()
        fp.seek(self.begin)

    def __next__(self):
        """Read the next frame."""
        if self.fp.tell() < self.end:
            return LvxFrame.read_from(self.fp)
        else:
            raise StopIteration()


class LvxFileReader(Iterable):
    """Handles reading a .lvx file."""

    def __init__(self, fp: BinaryIO):
        """Create a new LvxFileReader reading form fp."""
        fp.seek(0, 2)
        self.end = fp.tell()
        fp.seek(0)

        self.fp = fp
        self.header = LvxHeader.read_from(fp)

    def __iter__(self):
        """Get an iterator over all frames."""
        return LvxFrameIter(self.fp)

    @classmethod
    def make_csv(cls, fp: BinaryIO, s_dir: str, save_file: str, iterator: LvxFileReader, start_frame: int, end_frame: int):
        """Takes .lvx file path and saves relevant data in CSV"""

        if not os.path.isdir(s_dir):
            os.mkdir(s_dir)

        #TODO: Handle more than 6,7 datatypes
        #data_type_set = set()
        
        df_array = {
            "7": pd.DataFrame(columns=["x1", "y1", "z1", "reflectivity1", "tag1", "x2", "y2", "z2", "reflectivity2", "tag2", "x3", "y3", "z3", "reflectivity3", "tag3"]),
            "6": pd.DataFrame(columns=["gyro_x", "gyro_y", "gyro_z", "acc_x", "acc_y", "acc_z"])
        }

        for i, frame in enumerate(iterator):
            if i < start_frame:
                print(i)
            else:
                print(i)
                if i >= end_frame:
                    break
                for pack in frame.packages:
                    print(int.from_bytes(pack.timestamp, "little"))
                    if pack.data_type == 6:
                        for point in pack.points:
                            row_json = point.to_json()
                            df_array['6'].loc[len(df_array['6'])] = row_json
                    elif pack.data_type == 7:
                        for point in pack.points:
                            row_json = point.to_json()
                            df_array['7'].loc[len(df_array['7'])] = row_json
        
        for key in df_array:
            df_array[key].to_csv(s_dir+ "/" + key + save_file)
        
class LvxFileWriter:
    """Handles writing a .lvx file."""

    def __init__(self, header: LvxHeader, fp: BinaryIO):
        """Create a new LvxFileWriter writing to fp."""
        self.fp = fp
        self.header = header

        header.write_to(fp)

    def write_frame(self, frame: LvxFrame, fix_offset: bool = False):
        """Write frame to the file and optionally fix the header offsets."""
        if fix_offset:
            header = frame.header
            curr = self.fp.tell()
            diff = header.next_offset - header.current_offset
            header.current_offset = curr
            header.next_offset = curr + diff

        frame.write_to(self.fp)
test = LvxImuPoint()
print(test.size)