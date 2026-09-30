#!/usr/bin/env python3
"""Mark ML Kit's device arm64 slices as iOS-simulator arm64.

Google's ML Kit pods ship a fat binary whose arm64 slice is built for iOS
devices and whose x86_64 slice is the simulator. Apple Silicon simulators
link arm64, so the link fails. This rewrites LC_BUILD_VERSION in place
(platform IOS -> IOSSIMULATOR). When a slice has no platform command, a
24-byte LC_BUILD_VERSION is inserted into the padding before the first
section, which these object files have.

Idempotent. Safe to run after every `pod install`.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

FAT_MAGIC = 0xCAFEBABE
MH_MAGIC_64 = 0xFEEDFACF
CPU_TYPE_ARM64 = 0x0100000C
LC_BUILD_VERSION = 0x32
LC_SEGMENT_64 = 0x19
LC_SYMTAB = 0x2
LC_DATA_IN_CODE = 0x29
LC_LINKER_OPTIMIZATION_HINT = 0x2E
PLATFORM_IOS = 2
PLATFORM_IOSSIMULATOR = 7
# 12.0 / 18.0, matching the other ML Kit objects.
MINOS = 12 << 16
SDK = 18 << 16
BUILD_VERSION_SIZE = 24

ROOT = Path(__file__).resolve().parents[1]
FRAMEWORKS = [
    ROOT / "ios/Pods/MLImage/Frameworks/MLImage.framework/MLImage",
    ROOT / "ios/Pods/MLKitCommon/Frameworks/MLKitCommon.framework/MLKitCommon",
    ROOT / "ios/Pods/MLKitVision/Frameworks/MLKitVision.framework/MLKitVision",
    ROOT / "ios/Pods/MLKitFaceDetection/Frameworks/MLKitFaceDetection.framework/MLKitFaceDetection",
]


def pack_build_version() -> bytes:
    return struct.pack(
        "<IIIIII",
        LC_BUILD_VERSION,
        BUILD_VERSION_SIZE,
        PLATFORM_IOSSIMULATOR,
        MINOS,
        SDK,
        0,
    )


def patch_macho(buf: bytearray, start: int, size: int) -> int:
    if size < 32 or struct.unpack_from("<I", buf, start)[0] != MH_MAGIC_64:
        return 0
    cputype = struct.unpack_from("<I", buf, start + 4)[0]
    if cputype != CPU_TYPE_ARM64:
        return 0
    ncmds, sizeofcmds = struct.unpack_from("<II", buf, start + 16)
    end_cmds = start + 32 + sizeofcmds
    if end_cmds > start + size:
        raise RuntimeError("load commands extend past the slice")

    off = start + 32
    min_content = start + size
    found = False
    for _ in range(ncmds):
        cmd, cmdsize = struct.unpack_from("<II", buf, off)
        if cmd == LC_BUILD_VERSION and cmdsize >= 24:
            platform = struct.unpack_from("<I", buf, off + 8)[0]
            if platform == PLATFORM_IOS:
                struct.pack_into("<I", buf, off + 8, PLATFORM_IOSSIMULATOR)
                return 1
            return 0
        if cmd == LC_SEGMENT_64:
            nsects = struct.unpack_from("<I", buf, off + 64)[0]
            so = off + 72
            for _s in range(nsects):
                section_offset = struct.unpack_from("<I", buf, so + 48)[0]
                if section_offset:
                    min_content = min(min_content, start + section_offset)
                so += 80
        elif cmd == LC_SYMTAB:
            symoff, _, stroff, _ = struct.unpack_from("<IIII", buf, off + 8)
            if symoff:
                min_content = min(min_content, start + symoff)
            if stroff:
                min_content = min(min_content, start + stroff)
        elif cmd in (LC_DATA_IN_CODE, LC_LINKER_OPTIMIZATION_HINT):
            dataoff = struct.unpack_from("<I", buf, off + 8)[0]
            if dataoff:
                min_content = min(min_content, start + dataoff)
        off += cmdsize

    slack = min_content - end_cmds
    if slack < BUILD_VERSION_SIZE:
        raise RuntimeError(
            f"no room to insert LC_BUILD_VERSION (slack {slack} bytes at {start})"
        )
    buf[end_cmds : end_cmds + BUILD_VERSION_SIZE] = pack_build_version()
    struct.pack_into("<II", buf, start + 16, ncmds + 1, sizeofcmds + BUILD_VERSION_SIZE)
    return 1


def patch_ar(buf: bytearray, start: int, size: int) -> int:
    if buf[start : start + 8] != b"!<arch>\n":
        return 0
    changed = 0
    p = start + 8
    end = start + size
    while p + 60 <= end:
        raw_size = bytes(buf[p + 48 : p + 58]).strip()
        if not raw_size:
            break
        member_size = int(raw_size)
        data_off = p + 60
        name = bytes(buf[p : p + 16])
        content_off = data_off
        content_size = member_size
        if name.startswith(b"#1/"):
            name_len = int(bytes(name[3:]).strip() or b"0")
            content_off += name_len
            content_size -= name_len
        if content_size >= 4:
            magic = struct.unpack_from("<I", buf, content_off)[0]
            if magic == MH_MAGIC_64:
                changed += patch_macho(buf, content_off, content_size)
        nxt = data_off + member_size
        if nxt % 2:
            nxt += 1
        if nxt <= p:
            break
        p = nxt
    return changed


def patch_fat(buf: bytearray) -> int:
    if len(buf) < 8 or struct.unpack_from(">I", buf, 0)[0] != FAT_MAGIC:
        return patch_ar(buf, 0, len(buf)) or patch_macho(buf, 0, len(buf))
    nfat = struct.unpack_from(">I", buf, 4)[0]
    changed = 0
    off = 8
    for _ in range(nfat):
        cputype, _sub, offset, size, _align = struct.unpack_from(">IIIII", buf, off)
        off += 20
        if cputype != CPU_TYPE_ARM64:
            continue
        changed += patch_ar(buf, offset, size)
        if changed == 0:
            changed += patch_macho(buf, offset, size)
        # An archive reports its own count; don't also treat it as a macho.
        if buf[offset : offset + 8] == b"!<arch>\n":
            pass
        elif struct.unpack_from("<I", buf, offset)[0] == MH_MAGIC_64 and changed == 0:
            changed += patch_macho(buf, offset, size)
    return changed


def patch_file(path: Path) -> int:
    data = bytearray(path.read_bytes())
    if len(data) >= 8 and struct.unpack_from(">I", data, 0)[0] == FAT_MAGIC:
        nfat = struct.unpack_from(">I", data, 4)[0]
        changed = 0
        off = 8
        for _ in range(nfat):
            cputype, _sub, offset, size, _align = struct.unpack_from(">IIIII", data, off)
            off += 20
            if cputype != CPU_TYPE_ARM64:
                continue
            if data[offset : offset + 8] == b"!<arch>\n":
                changed += patch_ar(data, offset, size)
            else:
                changed += patch_macho(data, offset, size)
    else:
        changed = patch_fat(data)
    if changed:
        path.write_bytes(data)
    return changed


def main() -> int:
    missing = [p for p in FRAMEWORKS if not p.exists()]
    if missing:
        print("ML Kit frameworks not installed yet; skipping simulator patch.")
        return 0
    total = 0
    for path in FRAMEWORKS:
        n = patch_file(path)
        total += n
        print(f"{path.name}: patched {n} arm64 object(s)")
    print(f"done, {total} object(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
