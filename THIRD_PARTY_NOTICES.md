<!-- SPDX-License-Identifier: Apache-2.0 -->
# Third-party notices

Most of this repository is licensed under Apache-2.0. The following file has
different terms:

| File | Origin | License |
|---|---|---|
| `probes/cudnn_frost/prefill_fp8qk_sm120.py` | NVIDIA cudnn-frontend 1.27, `prefill_f16_sm120.py` | MIT |

`origin-private` is a privacy-preserving provenance alias for an implementation
previously maintained in an unpublished source tree. It records lineage without
identifying that tree, its account, or its operators.

## MIT permission notice

Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Techniques cited, not copied

SageAttention, FlashAttention 3 (FA3), and FlashInfer PR #4502 are cited as
technical context and comparison points. No source from those works is copied
into this repository unless it is separately identified above.
