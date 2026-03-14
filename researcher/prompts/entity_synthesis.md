You are writing a technical summary for a research report about "{topic}".

**Entity type:** {entity_type}
**Entity name:** {entity_name}
**Confidence score:** {confidence_score}/5 (based on {source_count} independent sources)

**Evidence — claims from the research corpus:**
{claims}

**Known contradictions:**
{contradictions}

---

Write a 2–4 paragraph technical summary of everything the research found about **{entity_name}**.

Guidelines:
- Be specific: cite numbers, benchmarks, VRAM requirements, version names
- Lead with the strongest, best-supported findings
- If contradictions exist, address them directly — explain the discrepancy if possible
- Note any important limitations, caveats, or conditions under which findings apply
- Do NOT pad with generic statements — every sentence should carry specific information
- Style: factual, direct, technical audience

Output plain prose only — no headers, no bullet points, no JSON.
