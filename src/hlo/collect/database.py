"""SQLite schema for videos, comments (with reply threading) and incremental search state."""
from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, Text, create_engine, inspect, text
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

from hlo.config import DB_URL, ensure_dirs

ensure_dirs()
engine = create_engine(DB_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def _utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Video(Base):
    __tablename__ = "videos"

    video_id = Column(String, primary_key=True)
    title = Column(String)
    description = Column(Text)
    channel_id = Column(String)
    published_at = Column(DateTime)
    keyword_matched = Column(String)
    first_seen_at = Column(DateTime, default=_utcnow)
    comments_disabled = Column(Boolean, default=False)
    comments_collected_at = Column(DateTime, nullable=True)  # set once every comment page was fetched

    comments = relationship("Comment", back_populates="video")


class Comment(Base):
    __tablename__ = "comments"

    comment_id = Column(String, primary_key=True)
    video_id = Column(String, ForeignKey("videos.video_id"))
    parent_id = Column(String, nullable=True)  # top-level comment id for replies
    author_hash = Column(String)  # SHA-256 of display name; raw names are never stored
    text = Column(Text)
    like_count = Column(Integer, default=0)  # likes at collection time
    published_at = Column(DateTime)
    last_updated_at = Column(DateTime)

    video = relationship("Video", back_populates="comments")


class CollectionState(Base):
    __tablename__ = "collection_state"

    keyword = Column(String, primary_key=True)
    last_search_time = Column(DateTime)


def init_db():
    Base.metadata.create_all(bind=engine)
    # databases created before comments_collected_at existed
    if "comments_collected_at" not in {c["name"] for c in inspect(engine).get_columns("videos")}:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE videos ADD COLUMN comments_collected_at DATETIME"))
    with engine.begin() as conn:
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_comments_video_id ON comments (video_id)"))
