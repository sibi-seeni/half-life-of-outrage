"""Collect all comments and replies for discovered videos (commentThreads.list / comments.list, 1 unit per call).

Only videos published on or before the study end date are visited, and comments or replies posted
after it are skipped (hlo.config.END_DATE, default 2026-04-30). Like counts are as of collection time.
"""
import hashlib
import os
import time
from datetime import datetime

from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

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


def collect_comments():
    youtube = get_youtube_client()
    db = SessionLocal()
    try:
        videos = (db.query(Video)
                  .filter(Video.comments_disabled.is_(False), Video.published_at <= end_datetime())
                  .all())
        for video in videos:
            print(f"Collecting: {video.title}")
            try:
                collect_video(youtube, db, video)
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
