import json
from datetime import datetime

import pytest
from googleapiclient.errors import HttpError

import hlo.collect.api as api
from hlo.collect.discovery import recent_windows


class _Resp(dict):
    def __init__(self, status):
        super().__init__()
        self.status = status
        self.reason = ""


def _err(status, reason):
    body = json.dumps({"error": {"errors": [{"reason": reason}]}}).encode()
    return HttpError(_Resp(status), body)


class _Req:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def execute(self):
        self.calls += 1
        out = self.outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


def test_retries_per_minute_rate_limit(monkeypatch):
    monkeypatch.setattr(api.time, "sleep", lambda s: None)
    req = _Req([_err(429, "rateLimitExceeded"), _err(403, "rateLimitExceeded"), {"items": []}])
    assert api.execute(req) == {"items": []}
    assert req.calls == 3


def test_daily_quota_stops_cleanly(monkeypatch):
    monkeypatch.setattr(api.time, "sleep", lambda s: None)
    with pytest.raises(api.DailyQuotaExhausted):
        api.execute(_Req([_err(403, "quotaExceeded")]))


def test_other_errors_propagate():
    with pytest.raises(HttpError):
        api.execute(_Req([_err(404, "videoNotFound")]))


def test_recent_windows_cover_range_without_gaps():
    w = recent_windows(datetime(2025, 1, 1), datetime(2026, 4, 30, 23, 59, 59))
    assert w[0][0] == datetime(2025, 1, 1) and w[-1][1] == datetime(2026, 4, 30, 23, 59, 59)
    assert all(a[1] == b[0] for a, b in zip(w, w[1:]))
    assert len(w) == 17
