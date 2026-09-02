<!-- SPDX-License-Identifier: Apache-2.0 -->
# Research notes

Tracked, free-form markdown for ideas that are not yet work: kernel concepts,
numerical tricks, quantization schemes, competing-design analyses. The rest of
the repository is deliberately rigid — records immutable, digests enforced,
claims bound to evidence. This folder is the one place designed for thinking
out loud, and committing a note here gives an idea a public timestamp without
pretending it is a result.

The contract below is the price of being tracked. It is light on purpose, but
the required header is not optional: it is what lets a stranger (or an agent)
triage fifty notes without reading them all.

## The contract

One idea per file, `research/<kebab-case-slug>.md`, starting with this header:

```markdown
- **Kind:** new-operator | new-implementation | modification | study | other
- **Status:** idea | explored | probed | adopted | retired
- **Author(s):** <GitHub handle(s)>
- **Opened:** YYYY-MM-DD · **Last updated:** YYYY-MM-DD
- **Numerics impact (best guess):** none | bit-visible | semantics-changing
- **Would touch (if realized):** <contract sections, files, version axes>
```

Copy [`TEMPLATE.md`](TEMPLATE.md). Everything after the header is free form.

### Kind

- **new-operator** — a different mathematical/quantized operator than the
  [operator contract](../docs/OPERATOR_CONTRACT.md) defines: new quantization
  scheme, mask family, decode path, page semantics. Realizing it means a new
  `operator_contract_version` and its own oracle.
- **new-implementation** — the same operator through a different dataflow
  (e.g., a `wgmma` mainloop): a new `layout_version`.
- **modification** — a change to an existing kernel, preprocessing step, or
  transform. The numerics-impact field matters most here: say up front
  whether the goldens would survive.
- **study** — analysis or controlled measurement of something that already
  exists, ours or someone else's (a competing kernel, an upstream PR, a
  paper's claim on our geometry).
- **other** — tooling, harness, methodology, anything that fits none of the
  above.

### Status

`idea` (written down, nothing tried) → `explored` (napkin math, simulation,
or literature done) → `probed` (a committed probe or record exists — link it)
→ `adopted` (graduated into the contract, a kernel, or a release — link where)
or `retired` (killed; **say why**, the tombstone is the value). Update the
header in place; git history is the changelog. Retired notes are never
deleted.

### Rules

1. **A note is not evidence.** Nothing in `research/` is a claim; the claims
   register and promotion records stay canonical. Any number quoted here
   either links to a committed record or is explicitly marked *(unrecorded
   estimate)*. CI does not test prose, so the only defense against research
   notes quietly becoming folklore-claims is this rule.
2. **Name the cheapest decisive test.** Every note states the smallest
   experiment that could kill or confirm it — even "none known" is
   informative. Ideas graduate through `probes/` and then the
   [artifact lifecycle](../docs/ARTIFACT_LIFECYCLE.md); they do not skip from
   prose to release.
3. **State prior art honestly.** A related-work line ("SageAttention2 does X;
   this differs by Y" / "not searched yet") beats accidental rediscovery.
   Novelty claims without a search are marked as such.
4. **Public means public.** Commit an idea here only when you are comfortable
   with it being read, dated, and quoted. The upside is precedence: the git
   timestamp is your public record of when the idea existed.
5. The repository-wide exclusions ([`AGENTS.md`](../AGENTS.md)) apply
   unchanged: no third-party datasets, no activation captures, no
   credentials, no results whose raw samples were discarded.
6. Contributor mechanics (one note or coherent set per PR, SPDX header on
   every file) follow [`CONTRIBUTING.md`](../CONTRIBUTING.md). Discussing an
   idea in an issue first is welcome but not required — unlike artifacts,
   ideas need no namespace reservation.

### Relation to the rest of the tree

| Genre | Lives in | Mutability |
|---|---|---|
| Idea, hypothesis, design sketch | `research/` | living document, header kept current |
| Executable experiment + its record | `probes/` | code evolves; committed records immutable |
| Qualified artifact | `promotion/` | append-only |
| Claim with evidence | `upstream/CLAIMS.md` | edited only with its evidence |

The existing research-handoff brief
([`docs/TRANSFORM_RESEARCH_BRIEF_2026-09-01.md`](../docs/TRANSFORM_RESEARCH_BRIEF_2026-09-01.md))
predates this folder and stays where it is; notes written after it land here.
