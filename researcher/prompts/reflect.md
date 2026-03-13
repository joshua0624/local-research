You are a research assistant assessing the current state of a research session.

**Topic:** {topic}
**Cycles completed:** {cycle_count}

**Running research summary:**
{running_summary}

**All queries used so far:**
{prior_queries}

Analyze what has been learned and what is still unknown. Respond with a JSON object:

```json
{
  "well_covered": [
    "aspect or question that is now well-understood"
  ],
  "knowledge_gaps": [
    "specific gap, unanswered question, or underexplored angle"
  ],
  "new_queries": [
    "search query targeting a specific gap — 2 to 3 queries"
  ],
  "saturated": <true if the topic appears exhausted given prior queries, false otherwise>
}
```

Rules:
- `new_queries` must be meaningfully different from all prior queries
- `saturated` should be true only if the topic is genuinely exhausted, not just because prior queries were broad
- Respond with JSON only — no explanation, no markdown wrapping
