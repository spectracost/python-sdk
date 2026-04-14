# Spectracost Python SDK

See the full spectrum of your AI spend.

```python
from openai import OpenAI
from spectracost import instrument

client = instrument(
    OpenAI(),
    api_key="sprc_...",
    team="search",
    service="query-rewriting",
)

# Use exactly like a normal OpenAI client - now with cost tracking
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
)
```
