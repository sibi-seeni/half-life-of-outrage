"""Collect all comments and replies for discovered videos (commentThreads.list / comments.list, 1 unit per call).

Only videos published on or before the study end date are visited, and comments or replies posted
after it are skipped (hlo.config.END_DATE, default 2026-04-30). Like counts are as of collection time.

A video is marked comments_collected_at once all its pages are fetched, and later runs skip it, so
each day's quota goes to videos not yet collected. first_from/first_to put an upload-date range first.
"""
import hashlib
import os
import time
from datetime import datetime, timezone

from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from sqlalchemy import and_, case, exists

from hlo.collect.api import DailyQuotaExhausted, execute
from hlo.collect.database import Comment, SessionLocal, Video
from hlo.config import end_datetime

load_dotenv()
_FMT = "%Y-%m-%dT%H:%M:%SZ"


def get_youtube_client():
    return build("youtube", "v3", developerKey=os.environ["YOUTUBE_API_KEY"])


def anonymize_id(user_id):
    if not user_id:
        return "anonymous"
    return hashlib.sha256(user_id.encode("utf-8")).hexdigest()


def save_comment(db, video_id, snippet, comment_id, seen, parent_id=None, end=None):
    if comment_id in seen:
        return
    published = datetime.strptime(snippet["publishedAt"], _FMT)
    if end is not None and published > end:
        return
    seen.add(comment_id)
    if db.query(Comment).filter_by(comment_id=comment_id).first():
        return
    db.add(Comment(
        comment_id=comment_id,
        video_id=video_id,
        parent_id=parent_id,
        author_hash=anonymize_id(snippet.get("authorDisplayName")),
        text=snippet["textDisplay"],
        like_count=snippet.get("likeCount", 0),
        published_at=published,
        last_updated_at=datetime.strptime(snippet["updatedAt"], _FMT),
    ))


def _all_replies(youtube, parent_id):
    """commentThreads only embeds up to 5 replies; page through comments.list for the rest."""
    page_token = None
    while True:
        response = execute(youtube.comments().list(
            part="snippet", parentId=parent_id, maxResults=100,
            pageToken=page_token, textFormat="plainText",
        ))
        yield from response.get("items", [])
        page_token = response.get("nextPageToken")
        if not page_token:
            return


def collect_video(youtube, db, video):
    end = end_datetime()
    seen = set()
    page_token = None
    while True:
        response = execute(youtube.commentThreads().list(
            part="snippet,replies", videoId=video.video_id, maxResults=100,
            order="time", pageToken=page_token, textFormat="plainText",
        ))
        time.sleep(0.1)

        for item in response.get("items", []):
            top = item["snippet"]["topLevelComment"]
            save_comment(db, video.video_id, top["snippet"], top["id"], seen, end=end)

            embedded = item.get("replies", {}).get("comments", [])
            if item["snippet"].get("totalReplyCount", 0) > len(embedded):
                embedded = _all_replies(youtube, top["id"])
            for reply in embedded:
                save_comment(db, video.video_id, reply["snippet"], reply["id"], seen,
                             parent_id=top["id"], end=end)

        page_token = response.get("nextPageToken")
        if not page_token:
            return


def mark_collected_from_existing():
    """One-off for databases from before comments_collected_at: mark every video that has comments.

    The video a quota stop interrupted is marked too, so it may be incomplete; there is one per
    past run at most.
    """
    db = SessionLocal()
    try:
        has_comments = exists().where(Comment.video_id == Video.video_id)
        n = (db.query(Video)
             .filter(Video.comments_collected_at.is_(None), has_comments)
             .update({Video.comments_collected_at: datetime.now(timezone.utc).replace(tzinfo=None)},
                     synchronize_session=False))
        db.commit()
        print(f"Marked {n} videos with existing comments as collected.")
    finally:
        db.close()


def collect_comments(first_from=None, first_to=None):
    youtube = get_youtube_client()
    db = SessionLocal()
    try:
        query = db.query(Video).filter(Video.comments_disabled.is_(False),
                                       Video.comments_collected_at.is_(None),
                                       Video.published_at <= end_datetime())
        order = [Video.published_at]
        if first_from or first_to:
            in_range = and_(*([Video.published_at >= first_from] if first_from else []),
                            *([Video.published_at < first_to] if first_to else []))
            order.insert(0, case((in_range, 0), else_=1))
        videos = query.order_by(*order).all()
        print(f"{len(videos)} videos left to collect")
        for video in videos:
            print(f"Collecting: {video.published_at:%Y-%m-%d} {video.title}")
            try:
                collect_video(youtube, db, video)
                video.comments_collected_at = datetime.now(timezone.utc).replace(tzinfo=None)
                db.commit()
            except DailyQuotaExhausted:
                db.commit()  # keep what this video already gave; it is revisited next run
                print("Daily YouTube quota used up. Progress is saved; run the same command again tomorrow.")
                raise
            except HttpError as e:
                db.rollback()
                if "commentsDisabled" in str(e):
                    video.comments_disabled = True
                    db.commit()
                else:
                    print(f"Error on {video.video_id}: {e}")
    finally:
        db.close()
