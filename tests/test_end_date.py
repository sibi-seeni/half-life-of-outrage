import sqlite3
from datetime import datetime

import pandas as pd


def test_build_dataset_drops_rows_after_end_date(tmp_path, monkeypatch):
    monkeypatch.setenv("HLO_END_DATE", "2026-04-30")
    import importlib

    import hlo.config
    importlib.reload(hlo.config)
    import hlo.data.build_dataset as bd
    importlib.reload(bd)

    db = tmp_path / "t.db"
    with sqlite3.connect(db) as c:
        c.execute("CREATE TABLE videos (video_id, title, description, channel_id, published_at, keyword_matched)")
        c.execute("CREATE TABLE comments (comment_id, video_id, parent_id, author_hash, text, like_count, "
                  "published_at, last_updated_at)")
        c.executemany("INSERT INTO videos VALUES (?,?,?,?,?,?)", [
            ("v1", "Arc Raiders trailer", "d", "ch", "2026-04-01 10:00:00", "Arc Raiders"),
            ("v2", "Arc Raiders patch", "d", "ch", "2026-05-02 10:00:00", "Arc Raiders")])
        text = "this game looks really great honestly"
        c.executemany("INSERT INTO comments VALUES (?,?,?,?,?,?,?,?)", [
            ("c1", "v1", None, "a", text + " one", 0, "2026-04-30 23:59:00", "2026-04-30 23:59:00"),
            ("c2", "v1", None, "a", text + " two", 0, "2026-05-01 00:00:01", "2026-05-01 00:00:01"),
            ("c3", "v2", None, "a", text + " three", 0, "2026-05-03 00:00:00", "2026-05-03 00:00:00")])
    df, _, flow = bd.build(str(db))
    assert list(df["comment_id"]) == ["c1"]
    assert flow["db_comments"] == 3 and flow["raw_comments"] == 1 and flow["end_date"] == "2026-04-30"


def test_collector_skips_comments_after_end_date(monkeypatch, tmp_path):
    monkeypatch.setenv("HLO_DATA_DIR", str(tmp_path))
    import importlib

    import hlo.config
    importlib.reload(hlo.config)
    import hlo.collect.database as database
    importlib.reload(database)
    import hlo.collect.collector as collector
    importlib.reload(collector)

    database.init_db()
    db = database.SessionLocal()
    end = datetime(2026, 4, 30, 23, 59, 59)

    def snip(ts):
        return {"authorDisplayName": "x", "textDisplay": "t", "likeCount": 0, "publishedAt": ts, "updatedAt": ts}

    seen = set()
    collector.save_comment(db, "v", snip("2026-04-30T12:00:00Z"), "keep", seen, end=end)
    collector.save_comment(db, "v", snip("2026-05-01T00:00:00Z"), "drop", seen, end=end)
    db.commit()
    assert [c.comment_id for c in db.query(database.Comment).all()] == ["keep"]
    db.close()
