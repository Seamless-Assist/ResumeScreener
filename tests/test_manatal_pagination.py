"""Pagination termination tests for the Manatal fetchers.

Manatal uses DRF page-number pagination: requesting a page past the last one
returns 404 {"detail": "Invalid page."}.  When the total number of records is an
exact multiple of page_size, a "stop when the page isn't full" heuristic asks for
one page too many and blows up.  These tests pin the `next`-link behaviour.
"""
import json
import time

import httpx
import pytest

from sa_candidate_finder import manatal_candidates, manatal_jobs


class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
        self.headers = {}
        self.url = "https://api.manatal.com/open/v3/jobs/"

    @property
    def text(self):
        return json.dumps(self._payload)

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"Client error '{self.status_code}' for url '{self.url}'",
                request=httpx.Request("GET", self.url),
                response=self,
            )


@pytest.fixture(autouse=True)
def no_sleep_no_cache_write(monkeypatch):
    """Keep tests fast and stop them clobbering the real on-disk caches."""
    monkeypatch.setattr(time, "sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(json, "dump", lambda *_a, **_k: None)


def _exact_multiple_pager(records, page_size, calls):
    """GET stub: one full page of `records`, 404 'Invalid page.' for any page > 1."""

    def fake_get(url, headers=None, params=None, timeout=None):
        page = int((params or {}).get("page", 1))
        calls.append(page)
        if page == 1:
            return FakeResponse(
                200,
                {"count": len(records), "next": None, "previous": None, "results": records},
            )
        return FakeResponse(404, {"detail": "Invalid page."})

    return fake_get


def test_fetch_all_jobs_stops_when_last_page_is_exactly_full(monkeypatch):
    jobs = [{"id": i, "position_name": f"Job {i}"} for i in range(100)]
    calls = []
    monkeypatch.setattr(
        manatal_jobs.httpx, "get", _exact_multiple_pager(jobs, 100, calls)
    )

    result = manatal_jobs.fetch_all_jobs("token", page_size=100, force_refresh=True)

    assert len(result) == 100
    assert calls == [1], f"should not request a page past the last one, requested {calls}"


def test_fetch_all_candidates_stops_when_last_page_is_exactly_full(monkeypatch):
    people = [{"id": i, "full_name": f"Person {i}"} for i in range(100)]
    calls = []
    monkeypatch.setattr(
        manatal_candidates.httpx, "get", _exact_multiple_pager(people, 100, calls)
    )

    result = manatal_candidates.fetch_all_candidates("token", page_size=100)

    assert len(result) == 100
    assert calls == [1], f"should not request a page past the last one, requested {calls}"


def test_fetch_all_jobs_still_follows_next_link(monkeypatch):
    """A genuine multi-page response must still be walked to the end."""
    page1 = [{"id": i} for i in range(100)]
    page2 = [{"id": 100 + i} for i in range(7)]
    calls = []

    def fake_get(url, headers=None, params=None, timeout=None):
        page = int((params or {}).get("page", 1))
        calls.append(page)
        if page == 1:
            return FakeResponse(
                200,
                {
                    "count": 107,
                    "next": "https://api.manatal.com/open/v3/jobs/?page=2&page_size=100",
                    "previous": None,
                    "results": page1,
                },
            )
        if page == 2:
            return FakeResponse(
                200, {"count": 107, "next": None, "previous": "...", "results": page2}
            )
        return FakeResponse(404, {"detail": "Invalid page."})

    monkeypatch.setattr(manatal_jobs.httpx, "get", fake_get)

    result = manatal_jobs.fetch_all_jobs("token", page_size=100, force_refresh=True)

    assert len(result) == 107
    assert calls == [1, 2]
