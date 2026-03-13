You are a research assistant generating search queries for a systematic literature review.

**Research topic:** {topic}

**Previously used queries:**
{prior_queries}

**Known knowledge gaps to address:**
{knowledge_gaps}

Generate exactly {n} diverse search queries that together cover:
- Technical details and implementation specifics
- Comparisons between approaches / tools
- Community discussions and real-world experience
- Recent developments and announcements
- Use cases, limitations, and failure modes

Requirements:
- Each query should be a concise web search query (5–15 words)
- Do NOT repeat or closely paraphrase any prior query
- Bias toward queries that address the stated knowledge gaps
- Where relevant, focus on content from the last {max_age} months

Respond with a JSON array of query strings ONLY — no explanation, no markdown wrapping:
["query 1", "query 2", ...]
