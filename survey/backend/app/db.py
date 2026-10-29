import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path

SCHEMA_PATH = Path(__file__).parent / "dbschema.sql"

def db_path() -> str:
    return os.environ.get("SURVEY_DB", "survey/backend/survey.sqlite")

# contextmanager manages the db connection and closes resources automatically
# [before yield __enter__ after yield __exit__ => close resources]
@contextmanager
def get_conn():
    conn = sqlite3.connect(db_path())
    # use dict-like rows instead of tuples
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def init_db() -> None:
    schema = SCHEMA_PATH.read_text()
    with get_conn() as connection:
        connection.executescript(schema)
