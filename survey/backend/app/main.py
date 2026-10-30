import json
import os
import secrets
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from sqlite3 import Connection

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from app.db import get_conn, init_db
from app.plan import make_plan
from app.schemas import (
    Answer,
    Participant,
    ParticipantState,
    QuestionKey,
    UserAdjustment,
)

# survey/assets to hold manifest.json and exported images
ASSETS_DIR = Path(os.environ.get("SURVEY_ASSETS", Path(__file__).resolve().parents[2] / "assets"))

@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield

app = FastAPI(title="Personalized AWB survey", lifespan=lifespan)

# allows svelte dev server (Vite, port 5173) to call the API during development
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:5173"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

@lru_cache
def get_manifest() -> dict:
    """@lru_cache caches the manifest, reading the file only the first time"""
    return json.loads((ASSETS_DIR / "manifest.json").read_text())

def now() -> str:
    return datetime.now(UTC).isoformat()

def get_participant_row(conn: Connection, participant_id: str):
    row = conn.execute("SELECT * FROM participants WHERE id = ?", (participant_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Unknown participant")
    return row

def find_trial(row, trial_index: int) -> dict:
    plan = json.loads(row["plan_json"])
    if not 0 <= trial_index < len(plan):
        raise HTTPException(status_code=422, detail="current question is not in this participant's plan")
    return plan[trial_index]

@app.post('/api/participants', response_model=ParticipantState, status_code=201)
def create_participant(body: Participant):
    manifest = get_manifest()
    # cryptographically secure random seed
    seed = secrets.randbelow(2**31)
    plan = make_plan(manifest, seed)
    participant_id = uuid.uuid4().hex

    with get_conn() as conn:
        conn.execute("""INSERT INTO participants (id, language, user_agent, screen_width, screen_height, seed, plan_json, created_at) VALUES
                     (?,?,?,?,?,?,?,?)""",
                     (
                         participant_id, body.language, body.user_agent, body.screen_width, body.screen_height, seed, json.dumps(plan), now()
                     ))
    return ParticipantState(participant_id=participant_id, plan=plan, answered=[])

@app.get("api/participants/{participant_id}", response_model=ParticipantState, status_code=200)
def get_participant(participant_id: str):
    with get_conn() as conn:
        row = get_participant_row(conn, participant_id)
        answered = [
            QuestionKey(question_index=trial["trial_index"], question_type=trial["question_type"])
            for trial in conn.execute("SELECT trial_index, question_type FROM trials WHERE participant_id = ?",
                                      (participant_id,))
        ]
    return ParticipantState(participant_id=participant_id, plan=json.loads(row["plan_json"]), answered=answered)

@app.post("api/trials", status_code=201)
def submit_answer(body: Answer):
    with get_conn() as conn:
        row = get_participant_row(conn, body.participant_id)
        trial_json = find_trial(row, body.question_index)
        if body.chosen_variant not in trial_json["variant_order"]:
            raise HTTPException(422, "chosen_variant was not shown in this trial")
        # upsert: resubmit = replace
        conn.execute("""INSERT INTO trials 
            (participant_id, trial_index, sample_id, question_type, chosen_variant, response_time_ms, created_at)
            VALUES (?,?,?,?,?,?,?)
            ON CONFLICT (participant_id, trial_index, question_type) DO UPDATE SET
                     chosen_variant=excluded.chosen_variant,
                     response_time_ms=excluded.response_time_ms,
                     created_at=excluded.created_at""",
             (
                 body.participant_id, body.question_index, trial_json["sample_id"], body.question_type, body.chosen_variant
             )
        )
    return {"ok": True}

@app.post("api/adjustments", status_code=201)
def submit_adjustment(body: UserAdjustment):
    with get_conn() as conn:
        row = get_participant_row(conn, body.participant_id)
        trial = find_trial(row, body.question_index)
        if body.base_variant not in trial["variant_order"]:
            raise HTTPException(422, "submitted variant was not shown in this trial")
        conn.execute("""INSERT INTO adjustments
                     (participant_id, trial_index, sample_id, question_type, base_variant, final_t, final_q, confirmed_as_is, created_at)
                     VALUES (?,?,?,?,?,?,?,?,?)
                     ON CONFLICT (participant_id, trial_index, question_type) DO UPDATE SET
                        base_variant=excluded.base_variant,
                        final_t=excluded.final_t,
                        final_q=excluded.final_q,
                        confirmed_as_is=excluded.confirmed_as_is,
                        created_at=excluded.created_at""",
                     (
                         body.participant_id, body.question_index, trial["sample_id"], body.question_type,
                         body.base_variant, body.final_t, body.final_q, int(body.confirmed_as_is), now()
                     )
        )
    return {"ok": True}

@app.post("/api/participants/{participant_id}/finish")
def finish(participant_id: str):
    with get_conn() as conn:
        get_participant_row(conn, participant_id)
        conn.execute("UPDATE participants SET finished_at = ? WHERE id = ?", (now(), participant_id))
    return {"ok": True}

@app.get("/")
async def root():
    return { "message": "Hello World" }

# serve manifest.json and image folders
app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")
