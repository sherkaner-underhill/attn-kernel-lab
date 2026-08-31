<!-- SPDX-License-Identifier: Apache-2.0 -->
# Repository operating contract

The repository must remain reproducible from its tracked files and published
records alone.

- **Never provision a GPU instance without explicit permission.** Rented
  instances are billable; CPU validation and the local development tier cover
  routine correctness work.
- A number is not a fact until it has a target, a workload, a timing backend, and
  raw samples. Do not quote a figure from prose when a record exists.
- `promotion/releases/` is **immutable**. Add a new record; never edit one.
  Attestations are append-only and reference a manifest digest.
- `tools/tree_digest.py` is the only sanctioned producer of `source_tree_sha256`,
  and `tools/gen_workload.py` the only sanctioned expander of a schedule. Do not
  reimplement either inline.
- Run `python3 tools/validate_registry.py` and `python3 -m pytest -q` before
  committing. Both are CPU-only and take under two seconds.
- Performance numbers from a `development`-authority target are diagnostic. If
  you find yourself wanting to promote one, the answer is to rent the production
  target, not to relax the rule.
- Third-party datasets and real activation captures never enter this repository.

```bash
pip install -r requirements-dev.txt
python3 tools/validate_registry.py --verbose
python3 -m pytest -q
```
