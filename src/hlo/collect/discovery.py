"""Discover Arc Raiders videos with the YouTube Data API v3 (search.list, 100 quota units per call).

YouTube search is case-insensitive, so the query list is just the plain name and the hashtag;
the case variants in KEYWORDS are only used to check video titles.

Two passes per query:
  1. Backfill: walks backwards in 90-day windows from the saved cursor until BACKFILL_START.
  2. Recent: RECENT_START..end date in 30-day windows. Search returns at most ~500 results per
     query, so smaller windows find more videos. Finished windows are recorded and skipped on
     the next run, so a run cut short by the daily quota resumes where it stopped.
Nothing published after the study end date (hlo.config.END_DATE, default 2026-04-30) is searched.
Only videos in the Gaming category (id 20) whose *title* contains a keyword are kept.
"""
import os
from datetime import datetime, timedelta

from dotenv import load_dotenv
from googleapiclient.discovery import build

from hlo.collect.api import SEARCH_PAUSE_S, DailyQuotaExhausted, execute
from hlo.collect.database import CollectionState, SessionLocal, Video, _utcnow
from hlo.config import end_datetime

load_dotenv()

SEARCH_QUERIES = ["arc raiders", "#arcraiders"]
KEYWORDS = [
    "Arc Raiders", "ARC RAIDERS", "arc raiders", "Arc raiders",
    "#arcraiders", "#ArcRaiders", "#ARCRAIDERS", "#Arcraiders",
]
BACKFILL_START = datetime(2021, 12, 9)  # day before the reveal at The Game Awards 2021
RECENT_START = datetime.strptime(os.getenv("HLO_RECENT_START", "2025-01-01"), "%Y-%m-%d")
WINDOW_DAYS = 90
RECENT_WINDOW_DAYS = 30
GAMING_CATEGORY_ID = "20"
_FMT = "%Y-%m-%dT%H:%M:%SZ"


def get_youtube_client():
    return build("youtube", "v3", developerKey=os.environ["YOUTUBE_API_KEY"])


def title_matches(title):
    title_lower = title.lower()
    return any(k.lower() in title_lower for k in KEYWORDS)


def _search_end():
    return min(_utcnow(), end_datetime())


def recent_windows(start, end, days=RECENT_WINDOW_DAYS):
    windows, cur = [], start
    while cur < end:
        nxt = min(cur + timedelta(days=days), end)
        windows.append((cur, nxt))
        cur = nxt
    return windows


def _search_window(youtube, db, query, published_after, published_before, seen):
    """Page through one search window and add unseen, title-matching videos. Returns # added."""
    added = 0
    page_token = None
    while True:
        response = execute(youtube.search().list(
            q=query,
            part="snippet",
            type="video",
            order="date",
            videoCategoryId=GAMING_CATEGORY_ID,
            maxResults=50,
            pageToken=page_token,
            publishedAfter=published_after.strftime(_FMT),
            publishedBefore=published_before.strftime(_FMT),
        ), pause=SEARCH_PAUSE_S)

        for item in response.get("items", []):
            video_id = item["id"]["videoId"]
            snippet = item["snippet"]
            if not title_matches(snippet["title"]) or video_id in seen:
                continue
            seen.add(video_id)
            if db.query(Video).filter_by(video_id=video_id).first():
                continue
            db.add(Video(
                video_id=video_id,
                title=snippet["title"],
                description=snippet["description"],
                channel_id=snippet["channelId"],
                published_at=datetime.strptime(snippet["publishedAt"], _FMT),
                keyword_matched=query,
            ))
            added += 1

        page_token = response.get("nextPageToken")
        if not page_token:
            return added


def _state(db, key, default):
    state = db.query(CollectionState).filter_by(keyword=key).first()
    if not state:
        state = CollectionState(keyword=key, last_search_time=default)
        db.add(state)
        db.commit()
    return state


def search_new_videos():
    youtube = get_youtube_client()
    db = SessionLocal()
    seen = set()
    end = _search_end()
    try:
        for query in SEARCH_QUERIES:
            # Recent pass, window by window; a done-marker row is stored per finished window.
            for w_start, w_end in recent_windows(RECENT_START, end):
                key = f"recent:{query}:{w_start:%Y-%m-%d}:{w_end:%Y-%m-%d}"
                if db.query(CollectionState).filter_by(keyword=key).first():
                    continue
                n = _search_window(youtube, db, query, w_start, w_end, seen)
                db.add(CollectionState(keyword=key, last_search_time=_utcnow()))
                db.commit()
                print(f"'{query}' recent {w_start:%Y-%m-%d}..{w_end:%Y-%m-%d}: +{n}")

            # Backfill pass: one 90-day window further back per run.
            state = _state(db, f"backfill:{query}", RECENT_START)  # recent windows cover later dates
            cursor = min(state.last_search_time, end)
            if cursor <= BACKFILL_START:
                continue
            after = max(BACKFILL_START, cursor - timedelta(days=WINDOW_DAYS))
            n = _search_window(youtube, db, query, after, cursor, seen)
            state.last_search_time = after
            db.commit()
            print(f"'{query}' backfill {after:%Y-%m-%d}..{cursor:%Y-%m-%d}: +{n}")
    except DailyQuotaExhausted:
        db.commit()
        print("Daily YouTube quota used up. Progress is saved; run the same command again tomorrow.")
        raise
    finally:
        db.close()
