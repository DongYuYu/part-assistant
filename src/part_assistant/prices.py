"""Claude list prices, used by the agent (trace, budget) and by the gateway (metering,
which models are allowed).

Its own module with no imports, so the gateway image does not need the agent and its
retrieval packages just to read a price table.
"""

# USD per million tokens (Claude API list price, checked 2026-11-13).
PRICES = {"claude-opus-5-5": {"input": 4.00, "output": 20.00,
                              "cache_read": 0.20, "cache_write": 5.00}}
