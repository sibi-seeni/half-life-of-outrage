"""Discover Arc Raiders videos with the YouTube Data API v3 (search.list, 100 quota units per call).

Two passes per keyword:
  1. Backfill: walks backwards in 90-day windows from the last cursor until BACKFILL_START.
  2. Recent: everything published since RECENT_START.
Nothing published after the study end date (hlo.config.END_DATE, default 2026-04-30) is searched.
Only videos in the Gaming category (id 20) whose *title* contains a keyword are kept.
"""
import os
from datetime import datetime, timedelta

from dotenv import load_dotenv
from googleapiclient.discovery import build

from hlo.collect.database import CollectionState, SessionLocal, Video, _utcnow
from hlo.config import end_datetime

load_dotenv()

KEYWORDS = [
    "Arc Raiders", "ARC RAIDERS", "arc raiders", "Arc raiders",
    "#arcraiders", "#ArcRaiders", "#ARCRAIDERS", "#Arcraiders",
]
BACKFILL_START = datetime(2021, 12, 9)  # day before the reveal at The Game Awards 2021
RECENT_START = datetime(2025, 1, 1)
WINDOW_DAYS = 90
GAMING_CATEGORY_ID = "20"
_FMT = "%Y-%m-%dT%H:%M:%SZ"


def get_youtube_client():
    return build("youtube", "v3", developerKey=os.environ["YOUTUBE_API_KEY"])


def title_matches(title):
    title_lower = title.lower()
    return any(k.lower() in title_lower for k in KEYWORDS)


def _search_window(youtube, db, keyword, published_after, published_before, seen):
    """Page through one search window and add unseen, title-matching videos. Returns # added."""
    added = 0
    page_token = None
    while True:
        response = youtube.search().list(
            q=keyword,
            part="snippet",
            type="video",
            order="date",
            videoCategoryId=GAMING_CATEGORY_ID,
            maxResults=50,
            pageToken=page_token,
            publishedAfter=published_after.strftime(_FMT),
            publishedBefore=published_before.strftime(_FMT),
        ).execute()

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
                keyword_matched=keyword,
            ))
            added += 1

        page_token = response.get("nextPageToken")
        if not page_token:
            return added


def _search_end():
    return min(_utcnow(), end_datetime())


def search_new_videos():
    youtube = get_youtube_client()
    db = SessionLocal()
    seen = set()
    try:
        for keyword in KEYWORDS:
            state = db.query(CollectionState).filter_by(keyword=keyword).first()
            if not state:
                state = CollectionState(keyword=keyword, last_search_time=_search_end())
                db.add(state)
                db.commit()
            cursor = min(state.last_search_time, _search_end())

            backfill_after = max(BACKFILL_START, cursor - timedelta(days=WINDOW_DAYS))
            n_back = _search_window(youtube, db, keyword, backfill_after, cursor, seen)
            n_recent = _search_window(youtube, db, keyword, RECENT_START, _search_end(), seen)
            print(f"'{keyword}': +{n_back} backfill ({backfill_after:%Y-%m-%d}..{cursor:%Y-%m-%d}), +{n_recent} recent")

            # Move the backfill cursor back so the next run covers the previous 90-day window.
            state.last_search_time = backfill_after
            db.commit()
    finally:
        db.close()
