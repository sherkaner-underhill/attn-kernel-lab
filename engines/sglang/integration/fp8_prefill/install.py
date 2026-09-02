#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Install the ``fp8_prefill`` attention backend into a pinned SGLang tree.

    python3 install.py apply  --sglang /sgl-workspace/sglang
    python3 install.py status --sglang /sgl-workspace/sglang
    python3 install.py revert --sglang /sgl-workspace/sglang

``apply``:
  1. copies the backend package (``__init__.py``/``backend.py`` from this
     directory, ``quant.py`` and ``csrc/fp8_prefill_attn.cu`` from this
     repository's canonical ``src/attn_kernel_lab/``) into the tree at
     ``python/sglang/srt/layers/attention/fp8_prefill/`` (new directory,
     no conflicts);
  2. registers the backend factory in ``attention_registry.py``
     (marker-guarded append) and adds ``fp8_prefill`` to the hybrid-GDN
     Blackwell allowlist (marker-guarded line replace);
  3. adds ``"fp8_prefill"`` to ``ATTENTION_BACKEND_CHOICES`` in
     ``server_args.py`` (marker-guarded insert).

``revert`` restores both edited files from the backups ``apply`` left and
removes the package directory.  Both directions ``py_compile``-check every
touched file, refuse double application, and print sha256 digests.  The
server must be restarted for either direction to take effect.

The edit anchors are written against SGLang commit
1cf2b8c54d81802abc15dcf23a29b9cc687bc01e (the pin every published serving
result used).  On any other commit the installer refuses unless
``--allow-unpinned`` is given AND both anchors are still unique.
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import py_compile
import shutil
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[3]
PIN = "1cf2b8c54d81802abc15dcf23a29b9cc687bc01e"

SOURCES = {
    "__init__.py": HERE / "__init__.py",
    "backend.py": HERE / "backend.py",
    "quant.py": REPO / "src/attn_kernel_lab/quant.py",
    "csrc/fp8_prefill_attn.cu": REPO / "src/attn_kernel_lab/csrc/fp8_prefill_attn.cu",
}

BACKUP_SUFFIX = ".attn-kernel-lab-fp8-prefill-backup"
MARKER = "fp8_prefill backend (attn-kernel-lab"

REGISTRY_APPEND = '''

# === fp8_prefill backend (attn-kernel-lab; reversible install) ===
@register_attention_backend("fp8_prefill")
def create_fp8_prefill_backend(runner):
    if runner.use_mla_backend:
        raise ValueError("fp8_prefill does not support MLA models.")
    from sglang.srt.layers.attention.fp8_prefill import FP8PrefillAttnBackend

    # Forward the runner's workspace policy unchanged (stock flashinfer
    # factory behavior).
    return FP8PrefillAttnBackend(
        runner, init_new_workspace=runner.init_new_workspace
    )
'''

CHOICES_ANCHOR = '\nATTENTION_BACKEND_CHOICES = [\n'
CHOICES_NEW = ('\nATTENTION_BACKEND_CHOICES = [\n'
               '    "fp8_prefill",  # fp8_prefill backend (attn-kernel-lab)\n')

# hybrid-GDN Blackwell allowlist: fp8_prefill subclasses FlashInferAttnBackend,
# so everything the GDN wrapper needs from a flashinfer-family backend holds.
GDN_ALLOW_ANCHOR = '                    allowed = {"triton", "trtllm_mha", "flashinfer"}\n'
GDN_ALLOW_NEW = ('                    allowed = {"triton", "trtllm_mha", "flashinfer",'
                 ' "fp8_prefill"}  # fp8_prefill backend (attn-kernel-lab)\n')


def digest(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["apply", "revert", "status"])
    parser.add_argument("--sglang", required=True, type=pathlib.Path,
                        help="root of the SGLang git checkout")
    parser.add_argument("--allow-unpinned", action="store_true",
                        help="proceed on a commit other than the pin")
    args = parser.parse_args()

    srt = args.sglang / "python/sglang/srt"
    registry = srt / "layers/attention/attention_registry.py"
    server_args = srt / "server_args.py"
    pkg_dst = srt / "layers/attention/fp8_prefill"
    if not registry.is_file() or not server_args.is_file():
        print(f"install: {srt} does not look like an SGLang srt tree", file=sys.stderr)
        return 1

    head = subprocess.run(
        ["git", "-C", str(args.sglang), "rev-parse", "HEAD"],
        capture_output=True, text=True,
    ).stdout.strip()
    if head != PIN:
        msg = f"install: tree HEAD {head or '<no git>'} is not the pin {PIN}"
        if not args.allow_unpinned:
            print(msg + " (use --allow-unpinned to proceed)", file=sys.stderr)
            return 1
        print("WARNING: " + msg, file=sys.stderr)

    reg_text = registry.read_text()
    sa_text = server_args.read_text()

    if args.command == "status":
        print(f"registry: {'APPLIED' if MARKER in reg_text else 'clean'} sha256={digest(registry)[:16]}")
        print(f"server_args: {'APPLIED' if MARKER in sa_text else 'clean'} sha256={digest(server_args)[:16]}")
        print(f"package dir: {'present' if pkg_dst.is_dir() else 'absent'}")
        if pkg_dst.is_dir():
            for rel in SOURCES:
                p = pkg_dst / rel
                if p.is_file():
                    same = digest(p) == digest(SOURCES[rel])
                    print(f"  {rel}: sha256={digest(p)[:16]} "
                          f"({'matches' if same else 'DIFFERS FROM'} repo source)")
        return 0

    if args.command == "apply":
        if MARKER in reg_text or MARKER in sa_text or pkg_dst.exists():
            print("install: already applied (or partial); revert first", file=sys.stderr)
            return 1
        for rel, src in SOURCES.items():
            if not src.is_file():
                print(f"install: missing source {src}", file=sys.stderr)
                return 1
        if sa_text.count(CHOICES_ANCHOR) != 1:
            print("install: choices anchor not unique; refusing", file=sys.stderr)
            return 1
        if reg_text.count(GDN_ALLOW_ANCHOR) != 1:
            print("install: GDN allowlist anchor not unique; refusing", file=sys.stderr)
            return 1

        (pkg_dst / "csrc").mkdir(parents=True)
        for rel, src in SOURCES.items():
            shutil.copy2(src, pkg_dst / rel)

        for path in (registry, server_args):
            backup = path.with_name(path.name + BACKUP_SUFFIX)
            if not backup.exists():
                shutil.copy2(path, backup)
        registry.write_text(
            reg_text.replace(GDN_ALLOW_ANCHOR, GDN_ALLOW_NEW) + REGISTRY_APPEND
        )
        server_args.write_text(sa_text.replace(CHOICES_ANCHOR, CHOICES_NEW))
        py_compile.compile(str(registry), doraise=True)
        py_compile.compile(str(server_args), doraise=True)
        for rel in ("__init__.py", "backend.py", "quant.py"):
            py_compile.compile(str(pkg_dst / rel), doraise=True)
        print(f"applied: registry sha256={digest(registry)[:16]} "
              f"server_args sha256={digest(server_args)[:16]}")
        for rel in ("quant.py", "csrc/fp8_prefill_attn.cu"):
            print(f"  staged {rel}: sha256={digest(pkg_dst / rel)}")
        return 0

    # revert
    for path in (registry, server_args):
        backup = path.with_name(path.name + BACKUP_SUFFIX)
        if backup.exists():
            shutil.copy2(backup, path)
            backup.unlink()
            py_compile.compile(str(path), doraise=True)
            print(f"{path.name}: reverted sha256={digest(path)[:16]}")
        else:
            print(f"{path.name}: no backup; unchanged")
    if pkg_dst.exists():
        shutil.rmtree(pkg_dst)
        print("package dir: removed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
