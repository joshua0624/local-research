You are extracting structured entity mentions from a batch of research findings about "{topic}".

Below are {n} findings, indexed 0 to {n_minus_1}:

{findings}

---

Identify every **named entity** mentioned across these findings. Entity types:
- **model** — a specific AI/LLM model (e.g. "Qwen 2.5 72B", "Llama 3.3 70B", "DeepSeek-R1")
- **tool** — a specific software tool, framework, or platform (e.g. "Aider", "Continue.dev", "Ollama", "vLLM")
- **workflow** — a described pattern, technique, or approach (e.g. "context window stuffing", "multi-agent coding loop", "speculative decoding")
- **hardware** — a specific hardware platform or configuration (e.g. "M4 Max", "RTX 4090", "48GB VRAM")

For each entity, collect:
- All specific factual claims made about it across the findings
- The indices of the findings that mention it
- The source URLs from those findings

Rules:
- Include every named entity mentioned even once — corroboration is counted later across all batches
- Claims must be specific and factual — no vague statements like "works well"
- Each claim should be a complete, standalone sentence
- Normalize entity names (e.g. "llama-3.3-70b" and "Llama 3.3 70B" are the same entity)

Return a JSON array only, no prose:

[
  {
    "entity_type": "model",
    "name": "Qwen 2.5 72B",
    "claims": [
      "Achieves near-perfect tool-calling reliability in agentic coding tasks.",
      "Requires at least 48GB VRAM at FP8 quantization for full performance."
    ],
    "finding_indices": [0, 4, 12],
    "source_urls": ["https://..."]
  }
]

If no entities are found, return an empty array: []
