CREATE TABLE IF NOT EXISTS participants (
    id                  TEXT PRIMARY KEY, --random uuid [no personal data]
    language            TEXT NOT NULL,
    user_agent          TEXT,
    screen_width        INTEGER,
    screen_height       INTEGER,
    seed                INTEGER NOT NULL, --random plan generator
    plan_json           TEXT NOT NULL,
    created_at          TEXT NOT NULL,
    finished_at         TEXT
);

CREATE TABLE IF NOT EXISTS trials (
    id                  INTEGER PRIMARY KEY,
    participant_id      TEXT NOT NULL REFERENCES participants(id),
    trial_index         INTEGER NOT NULL,
    sample_id           TEXT NOT NULL,
    question_type       TEXT NOT NULL,
    chosen_variant      TEXT NOT NULL,  -- variant id = algorithm id
    response_time_ms    INTEGER NOT NULL,
    created_at          TEXT NOT NULL,
    UNIQUE (participant_id, trial_index, question_type)
);

CREATE TABLE IF NOT EXISTS adjustments (
    id                  INTEGER PRIMARY KEY,
    participant_id      TEXT NOT NULL REFERENCES participants(id),
    trial_index         INTEGER NOT NULL,
    sample_id           TEXT NOT NULL,
    question_type       TEXT NOT NULL,
    base_variant        TEXT NOT NULL,
    final_t             REAL NOT NULL,
    final_q             REAL NOT NULL,
    confirmed_as_is     INTEGER NOT NULL,
    created_at          TEXT NOT NULL,
    UNIQUE (participant_id, trial_index, question_type)
)