# Generative Citation Scouting — methodology digest

**Date:** 2026-07-17 · **For:** `scholar` `literature` capability, `scout` mode (sub-spec 2). · **Status:** verified-source digest; migrates to the plugin's `resources/references/`.

Using the citation graph to generate research ideas, spot gaps, and find collaborators, anchored on an author's paper + rival anchors. Identifiers verified at compile time unless flagged `[unverified]`.

## 1. Named methods

- **Forward-citation chasing / snowballing** — walk an anchor's *citing* set (forward) and *reference* set (backward), iterate with explicit inclusion criteria; forward surfaces "who built on us / the rivals." Rigorous protocol: Wohlin, C. (2014), *Guidelines for snowballing in systematic literature studies*, EASE '14, DOI:10.1145/2601248.2601268.
- **Citation context / function / sentiment** — classify *why* a paper cites another (background vs method-use vs result-comparison; supporting vs contrasting). Core primitive for hypothesis-level scouting. Teufel, Siddharthan, Tidhar (2006), *Automatic classification of citation function*, EMNLP, ACL W06-1613. Cohan et al. (2019), *Structural Scaffolds for Citation Intent Classification*, NAACL, arXiv:1904.01608 → **SciCite** (Background/Method/Result-comparison), the scheme the Semantic Scholar API exposes.
- **Co-citation analysis** — two papers related if a later paper cites both; clusters reveal a specialty's intellectual base / research fronts. Small, H. (1973), *JASIS* 24(4):265–269, DOI:10.1002/asi.4630240406.
- **Bibliographic coupling** — two papers related if they share references; works for brand-new papers. Kessler, M.M. (1963), *American Documentation* 14(1):10–25, DOI:10.1002/asi.5090140103.
- **Research-front detection / gap-finding** — the growing tip of a literature. de Solla Price (1965), *Science* 149(3683):510–515, DOI:10.1126/science.149.3683.510. Operationalized via co-citation clustering + burst detection: Chen (2006) CiteSpace II `[unverified]`; Kleinberg (2002) KDD burst model `[unverified]`.
- **Main-path analysis** — dominant knowledge-flow chain through a citation DAG (weights links, not nodes). Hummon & Doreian (1989), *Social Networks* 11:39–63.
- **Idea backlog composition** (not a single named method) — forward-chase → context-classify → cluster → tag each by idea type (untested extension, contradiction/rival, new domain, transferable technique, methodological gap). See §3.

## 2. Tooling (ranked for a scriptable, reproducible workflow)

| Tool | Free? | Scriptable API | Note |
|---|---|---|---|
| **OpenAlex** | yes, open | REST, **no key** (`mailto=` polite pool) | `referenced_works` + `filter=cites:WORKID`. **Best backbone.** arXiv:2205.01833 |
| **Semantic Scholar (S2AG)** | yes | REST; free key ↑ limit | `/paper/{id}/citations` returns **contexts + intents (SciCite)** + `isInfluential`; `recommendations` = idea expansion. S2ORC arXiv:1911.02782; API arXiv:2301.10140 |
| **Inciteful** | yes | API `[unverified]` | Paper Discovery + Literature Connector (path between two papers) |
| **Connected Papers** | freemium | no API | co-citation + coupling visual graph; research-front snapshot |
| **Litmaps / ResearchRabbit** | freemium/free | limited | seed-set expansion + monitoring alerts (collaborator/forward-cite watch) |
| **scite.ai** | paid | API `[unverified]` | Supporting/Mentioning/Contrasting — productized citation sentiment |
| **VOSviewer / CitNetExplorer** | free desktop | file-based | co-citation/coupling maps; main-path (paper-level map) `[unverified]` |
| **Google Scholar** | web | **no API** (ToS) | broadest coverage; sanity-check only, never the backbone |
| **Scopus / Web of Science** | subscription | API (institutional) | curated counts + venue metadata; no citation context |

**Recommendation:** OpenAlex backbone + Semantic Scholar for contexts/intents/recommendations. Both free, keyless-or-cheap, JSON, reproducible.

## 3. Process: anchor → ranked ideas + collaborators

1. Fix the anchor set (own papers + rival anchors); resolve each to a stable ID (DOI/arXiv/OpenAlex/S2).
2. Pull forward citations per anchor (OpenAlex `filter=cites:`, S2 `/citations`); store raw JSON = provenance root.
3. Enrich each citing paper: year, venue, citation count, abstract, authors+affiliations, **citation-context snippet + intent** (S2).
4. Filter/prioritize: intent (Method/Result-comparison > Background; `isInfluential`), recency (24–36 mo up for fronts), impact (tie-breaker, not primary), coupling proximity.
5. Cluster by co-citation + bibliographic coupling to expose sub-fronts.
6. Classify each cluster/paper into an idea type (extension / contradiction / new domain / transferable technique / gap).
7. Emit a ranked backlog row per idea with **mandatory provenance**: `idea | type | source paper (ID) | citing-context snippet | why-it-matters | est. feasibility`.
8. Collaborators: rank citing-paper authors by frequency × recency × non-overlap with existing coauthors.
9. Make it standing (Litmaps watch / scheduled OpenAlex query) → monitored feed.

**Provenance rule:** every generated idea carries the citing paper's ID + exact citation-context snippet — the auditable link from idea back to who-said-what.

## 4. Level split — same method, two tunings (not two methods)

- **Hypothesis level** (narrow): precision-optimized; small set, read full text, **citation-context/sentiment dominates** (find Contrasting/Result-comparison citations).
- **Paper level** (broad): recall-optimized; large set, skim metadata, **co-citation clustering + burst/front detection dominates**.

→ one pipeline with a `level` parameter swapping (i) ranking weights, (ii) reading depth, (iii) primary technique. Separate methods would duplicate the graph plumbing.

## 5. Idea-shaping frameworks (back end)

- **Sandberg & Alvesson (2011)**, *Ways of constructing research questions: gap-spotting or problematization?*, *Organization* 18(1):23–44, DOI:10.1177/1350508410372151 — flag mere gap-spotting vs assumption-challenging problematization.
- **Alon (2009)**, *How to choose a good scientific problem*, *Molecular Cell* 36(6):721–723, DOI:10.1016/j.molcel.2009.09.013 — feasibility × interest scoring.
- **Hamming, "You and Your Research"** (1986) `[unverified]` — filter against pure gap-filling.

## Sources
Small 1973 (asi.4630240406) · Kessler 1963 (asi.5090140103) · Hummon & Doreian 1989 · Teufel et al. 2006 (ACL W06-1613) · Cohan et al. 2019 (arXiv:1904.01608, SciCite) · Price 1965 (science.149.3683.510) · Sandberg & Alvesson 2011 (1350508410372151) · Alon 2009 (S1097-2765(09)00641-8) · S2AG API (arXiv:2301.10140), S2ORC (arXiv:1911.02782) · OpenAlex (arXiv:2205.01833) · Wohlin 2014 (10.1145/2601248.2601268).

## Flags
`[unverified this session]` (confident from domain knowledge): CiteSpace (Chen 2006 JASIST 57(3):359–377), Kleinberg 2002 KDD, VOSviewer (van Eck & Waltman 2010), CitNetExplorer (2014), scite.ai API, Inciteful API, Hamming 1986, Wohlin 2014 DOI.
