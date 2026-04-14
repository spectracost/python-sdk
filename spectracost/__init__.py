"""Spectracost — AI cost observability SDK.

See the full spectrum of your AI spend.

Usage:
    from openai import OpenAI
    from spectracost import instrument, attribution

    client = instrument(OpenAI(), api_key="sprc_...")

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello!"}],
    )

    # Per-call attribution overrides
    with attribution(feature="search", customer_id="cust_123"):
        response = client.chat.completions.create(...)
"""

from .instrument import instrument
from .attribution import attribution

__version__ = "0.1.1"
__all__ = ["instrument", "attribution"]
