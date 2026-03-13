You are evaluating whether a new research finding adds value beyond an existing one.

**Existing finding:**
{existing_finding}

**New finding:**
{new_finding}

Does the new finding:
(a) Contradict the existing finding?
(b) Update or supersede it with newer/more specific information?
(c) Add critical nuance or a meaningfully different angle?

If yes to any of (a), (b), or (c): keep both findings.
If the new finding is essentially a restatement or subset of the existing one: discard the new one.

Respond with JSON only:
{"keep": true, "reason": "brief explanation"}
or
{"keep": false, "reason": "brief explanation"}
