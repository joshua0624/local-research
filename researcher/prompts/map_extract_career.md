You are extracting structured entity mentions from a batch of research findings about "{topic}".

Below are {n} findings, indexed 0 to {n_minus_1}:

{findings}

---

Identify every **named entity** mentioned across these findings. Entity types:
- **skill** — a specific technical or soft skill relevant to career progression (e.g. "Kubernetes", "system design", "incident response", "Python", "on-call ownership", "technical writing")
- **tool** — a specific software tool or platform useful for an engineer's workflow or career (e.g. "GitHub Copilot", "Cursor", "Datadog", "Terraform", "PagerDuty", "Obsidian")
- **strategy** — a described career advancement approach, behavior, or pattern (e.g. "scope expansion", "leading post-mortems", "shadowing staff engineers", "building internal tools", "taking ownership of on-call")
- **resource** — a specific learning resource (e.g. "roadmap.sh", "Designing Data-Intensive Applications", "CNCF certifications", "The Staff Engineer's Path")

For each entity, collect:
- All specific factual claims or advice made about it across the findings
- The indices of the findings that mention it
- The source URLs from those findings

Rules:
- Include every named entity mentioned even once — corroboration is counted later across all batches
- Claims must be specific and actionable — no vague statements like "is useful" or "helps a lot"
- Each claim should be a complete, standalone sentence
- Normalize entity names (e.g. "k8s" and "Kubernetes" are the same entity)
- Do NOT invent entities not present in the findings

Return a JSON array only, no prose:

[
  {
    "entity_type": "skill",
    "name": "Kubernetes",
    "claims": [
      "Mid-level engineers are expected to debug pod scheduling failures without escalating.",
      "CKA certification is frequently cited as a credible signal during hiring."
    ],
    "finding_indices": [0, 4, 12],
    "source_urls": ["https://..."]
  }
]

If no entities are found, return an empty array: []
