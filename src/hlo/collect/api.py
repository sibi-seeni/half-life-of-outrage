"""YouTube API call wrapper: per-minute throttling, retry on rate limits, clean stop on daily quota."""
import json
import time

from googleapiclient.errors import HttpError

SEARCH_PAUSE_S = 1.0  # search.list has a per-minute cap; ~60 calls/min stays under it
RATE_LIMIT_WAIT_S = 65
MAX_RATE_LIMIT_RETRIES = 5


class DailyQuotaExhausted(RuntimeError):
    """Daily quota used up: progress is saved, rerun after the quota resets (midnight Pacific)."""


def _reason(err):
    try:
        return json.loads(err.content)["error"]["errors"][0].get("reason", "")
    except (ValueError, KeyError, IndexError, TypeError):
        return ""


def execute(request, pause=0.0):
    """Run a googleapiclient request. Waits and retries on per-minute rate limits,
    raises DailyQuotaExhausted when the daily quota is gone."""
    for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
        try:
            result = request.execute()
            if pause:
                time.sleep(pause)
            return result
        except HttpError as err:
            reason = _reason(err)
            if reason in ("quotaExceeded", "dailyLimitExceeded"):
                raise DailyQuotaExhausted(reason) from err
            if err.resp.status == 429 or reason in ("rateLimitExceeded", "userRateLimitExceeded"):
                if attempt == MAX_RATE_LIMIT_RETRIES:
                    raise
                print(f"  rate limited ({reason or err.resp.status}); waiting {RATE_LIMIT_WAIT_S}s")
                time.sleep(RATE_LIMIT_WAIT_S)
                continue
            raise
