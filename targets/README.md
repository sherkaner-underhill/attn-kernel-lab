<!-- SPDX-License-Identifier: Apache-2.0 -->
# Target profiles

One YAML file per GPU target; `id` must equal the filename stem. Validated
against `schema/target-profile.schema.json` by `tools/validate_registry.py`.

See `../docs/TARGETS.md` for what the fields mean, why the authority levels are
not a severity scale, and how to add a target.

Two fields carry more weight than their size suggests:

- **`authority`** decides whether this hardware can produce promotion evidence.
  It is checked mechanically, in both directions.
- **`verification.state`** records whether the numbers below it were read off the
  device or copied from a datasheet. `declared` is honest; a `production` target
  that is `unverified` is a validation error.
