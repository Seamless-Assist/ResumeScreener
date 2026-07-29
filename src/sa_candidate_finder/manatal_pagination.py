"""Shared pagination helpers for the Manatal REST API.

Manatal uses Django REST Framework page-number pagination.  Responses look like:

    {"count": 100, "next": null, "previous": null, "results": [...]}

Requesting a page past the last one returns HTTP 404 ``{"detail": "Invalid page."}``.
So callers must never speculatively ask for page N+1 — in particular, "the page was
full, so there is probably another one" is wrong whenever the total record count is
an exact multiple of ``page_size``.
"""
from typing import Any, List


def has_next_page(data: Any, results: List[Any], page_size: int) -> bool:
    """Return True when another page should be requested.

    Trusts the ``next`` link whenever the response provides one; falls back to the
    page-fullness heuristic only for payload shapes that omit it.
    """
    if isinstance(data, dict) and "next" in data:
        return bool(data.get("next"))
    return len(results) >= page_size


def is_invalid_page_response(status_code: int, page: int) -> bool:
    """True when a 404 just means "you walked past the last page"."""
    return status_code == 404 and page > 1
