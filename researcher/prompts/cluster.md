You are a research assistant organizing findings on the topic: "{topic}".

Below are {count} research findings, each prefixed with its index number:

{findings}

## Task

Group these findings into 3–6 thematic sections. Section names should be short (2–5 words) and descriptive (e.g. "Model Architecture", "Deployment & Tooling", "Benchmark Results").

Return ONLY a JSON object where:
- keys are finding index numbers as strings ("0", "1", "2", …)
- values are the section name for that finding

Assign every index. Keep related findings in the same section.

Example output format:
```json
{
  "0": "Model Architecture",
  "1": "Benchmark Results",
  "2": "Model Architecture",
  "3": "Deployment & Tooling"
}
```
