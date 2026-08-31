#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Read a GPU target's real properties, and check a profile's declared values.

A target profile carries a ``verification.state``: ``declared`` means a field was
taken from vendor documentation, ``measured`` means it was read off the device.
This tool is what moves a profile between those states, and what catches a
declared value that turns out to be wrong.

    probe_target.py                              # print what this device reports
    probe_target.py --check targets/<id>.yaml    # compare against a profile
    probe_target.py --compile-gate sm_120a       # can nvcc target another arch?

The shared-memory ceiling is the field this exists for. ``sharedMemPerBlock``
reports the 48 KB *default*, not the opt-in maximum a kernel can request, and the
production wide path needs ~98 KiB. Reading the wrong attribute would make a
target look unusable, or -- worse -- make an unusable one look fine.
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile

# cudaDeviceAttr values (driver/runtime stable).
ATTR_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97
ATTR_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 81


def _cudart() -> ctypes.CDLL | None:
    for candidate in ("libcudart.so.13", "libcudart.so.12", "libcudart.so"):
        try:
            return ctypes.CDLL(candidate)
        except OSError:
            continue
    found = ctypes.util.find_library("cudart")
    return ctypes.CDLL(found) if found else None


def _device_attribute(attr: int, device: int = 0) -> int | None:
    lib = _cudart()
    if lib is None:
        return None
    value = ctypes.c_int()
    if lib.cudaDeviceGetAttribute(ctypes.byref(value), ctypes.c_int(attr), ctypes.c_int(device)) != 0:
        return None
    return value.value


def probe(device: int = 0) -> dict:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device visible")

    props = torch.cuda.get_device_properties(device)
    optin = _device_attribute(ATTR_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, device)
    per_sm = _device_attribute(ATTR_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR, device)

    return {
        "raw_model": props.name,
        "compute_capability": [props.major, props.minor],
        "sm_count": props.multi_processor_count,
        "memory_gib": round(props.total_memory / 2**30, 2),
        "max_shared_memory_per_block_default_bytes": props.shared_memory_per_block,
        "max_shared_memory_per_block_bytes": optin,
        "max_shared_memory_per_multiprocessor_bytes": per_sm,
        "l2_cache_bytes": getattr(props, "L2_cache_size", None),
        "regs_per_multiprocessor": getattr(props, "regs_per_multiprocessor", None),
        "toolchain": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "nvcc": _nvcc_version(),
        },
    }


def _nvcc_version() -> str | None:
    nvcc = shutil.which("nvcc")
    if not nvcc:
        return None
    out = subprocess.run([nvcc, "--version"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if "release" in line:
            return line.strip()
    return None


# A trivial kernel compiles for almost any architecture and therefore gates
# nothing. This source contains an ARCHITECTURE-SPECIFIC instruction -- the
# SM120a block-scaled MMA -- so a successful build is evidence the toolchain can
# actually emit code for the target, and a failure is the real failure.
COMPILE_GATE_SOURCE = """
#include <cstdio>
#include <cstdint>

__global__ void probe_kernel(float* out, const uint32_t* a, const uint32_t* b,
                             const uint32_t* sfa, const uint32_t* sfb) {
#if defined(ARCH_SPECIFIC_PROBE)
  float d[4] = {0.f, 0.f, 0.f, 0.f};
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale."
      "scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "
      "{%10}, {0, 0}, {%11}, {0, 0};\\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
        "r"(b[0]), "r"(b[1]), "r"(sfa[0]), "r"(sfb[0]));
  out[threadIdx.x] = d[0] + d[1] + d[2] + d[3];
#else
  out[threadIdx.x] = float(threadIdx.x);
#endif
}

int main() { printf("ok\\n"); return 0; }
"""


def _wants_block_scale_probe(arch: str) -> bool:
    """Block-scaled MMA exists from SM120 onward; probing for it anywhere else
    would fail for the right reason and the wrong purpose."""
    digits = arch.removeprefix("sm_").rstrip("af")
    return digits.isdigit() and int(digits) >= 120


def compile_gate(arch: str, *, bare_arch: bool = False, arch_specific: bool | None = None) -> dict:
    """Can nvcc generate code for ``arch`` on this machine?

    Cross-compilation does not need the device present, which is what lets the
    development tier gate production-architecture code on every commit.

    ``arch_specific`` defaults to whether ``arch`` has block-scaled MMA at all.
    A trivial kernel compiles almost anywhere and therefore gates nothing, but a
    block-scale probe aimed at SM89 would fail for a reason that says nothing
    about the toolchain.

    ``bare_arch`` reproduces the recorded trap: ``-arch=sm_120a`` on an
    executable build silently lowers to ``sm_120``, after which ptxas rejects
    architecture-specific instructions. Always prefer the explicit
    ``-gencode=arch=compute_Xa,code=sm_Xa`` form.
    """
    nvcc = shutil.which("nvcc")
    if not nvcc:
        return {"arch": arch, "available": False, "reason": "nvcc not on PATH"}

    if arch_specific is None:
        arch_specific = _wants_block_scale_probe(arch)

    flag = [f"-arch={arch}"] if bare_arch else [
        f"-gencode=arch=compute_{arch.removeprefix('sm_')},code={arch}"
    ]
    if arch_specific:
        flag.append("-DARCH_SPECIFIC_PROBE=1")
    with tempfile.TemporaryDirectory() as tmp:
        src = pathlib.Path(tmp) / "probe.cu"
        src.write_text(COMPILE_GATE_SOURCE)
        result = subprocess.run(
            [nvcc, *flag, "-o", str(pathlib.Path(tmp) / "probe"), str(src)],
            capture_output=True,
            text=True,
        )
    return {
        "arch": arch,
        "probe": "block_scale_mma" if arch_specific else "trivial",
        "flags": flag,
        "available": result.returncode == 0,
        "returncode": result.returncode,
        "stderr": result.stderr.strip()[:500],
    }


CHECKED_FIELDS = (
    ("device", "raw_model", "raw_model"),
    ("device", "compute_capability", "compute_capability"),
    ("device", "sm_count", "sm_count"),
    ("device", "memory_gib", "memory_gib"),
    ("capabilities", "max_shared_memory_per_block_bytes", "max_shared_memory_per_block_bytes"),
)


def check(profile_path: pathlib.Path, measured: dict) -> list[str]:
    import yaml

    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    mismatches = []
    for section, key, probed_key in CHECKED_FIELDS:
        declared = profile.get(section, {}).get(key)
        actual = measured.get(probed_key)
        if declared is None or actual is None:
            continue
        if isinstance(declared, float) or isinstance(actual, float):
            ok = abs(float(declared) - float(actual)) < 0.05
        else:
            ok = declared == actual
        if not ok:
            mismatches.append(f"{section}.{key}: profile says {declared!r}, device reports {actual!r}")
    return mismatches


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--check", type=pathlib.Path, help="target profile to compare against")
    parser.add_argument("--compile-gate", metavar="ARCH", help="e.g. sm_120a")
    parser.add_argument("--bare-arch", action="store_true", help="use -arch instead of -gencode")
    parser.add_argument(
        "--trivial",
        action="store_true",
        help="compile a plain kernel instead of the block-scaled-MMA probe",
    )
    args = parser.parse_args(argv)

    if args.compile_gate:
        result = compile_gate(
            args.compile_gate,
            bare_arch=args.bare_arch,
            arch_specific=False if args.trivial else None,
        )
        json.dump(result, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0 if result["available"] else 1

    measured = probe(args.device)
    json.dump(measured, sys.stdout, indent=2)
    sys.stdout.write("\n")

    if args.check:
        mismatches = check(args.check, measured)
        if mismatches:
            print(f"\n{args.check}: {len(mismatches)} mismatch(es)", file=sys.stderr)
            for line in mismatches:
                print(f"  {line}", file=sys.stderr)
            return 1
        print(f"\n{args.check}: declared values match the device")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
