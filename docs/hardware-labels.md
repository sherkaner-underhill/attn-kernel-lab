<!-- SPDX-License-Identifier: Apache-2.0 -->
# Public hardware labels

Raw GPU UUIDs are intentionally omitted. Each label is stable within this
repository, so records carrying different labels were produced on different
physical devices.

| Label | Public description |
|---|---|
| `card-A` | RTX PRO 6000 Blackwell device used for allocation 1 measurements |
| `card-B` | RTX PRO 6000 Blackwell device used for allocation 2 confirmation |
| `card-C` | RTX PRO 6000 Blackwell device used for the CUDA Graph schedule |
| `local-dev-gpu` | Local GeForce RTX 4090 development device |

These aliases preserve the distinct-device evidence without publishing stable
hardware identifiers.
