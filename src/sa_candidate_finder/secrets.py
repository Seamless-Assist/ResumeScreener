import os
from typing import Final

# Runtime secrets are loaded from environment variables.
# Example (PowerShell):
#   $env:OPENAI_API_KEY="..."
#   $env:MANATAL_API_KEY="..."
#   $env:MANATAL_BASE_URL="https://mcp.manatal.com/p/..."
#   $env:GOODFIT_API_KEY="..."
OPENAI_API_KEY: Final[str] = os.getenv("OPENAI_API_KEY", "")
MANATAL_API_KEY: Final[str] = os.getenv("MANATAL_API_KEY", "")
MANATAL_BASE_URL: Final[str] = os.getenv("MANATAL_BASE_URL", "")
GOODFIT_API_KEY: Final[str] = os.getenv("GOODFIT_API_KEY", "")
