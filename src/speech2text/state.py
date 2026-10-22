import fcntl
import hashlib
import json
import os
import sqlite3
import tempfile
from contextlib import contextmanager
from pathlib import Path

from .config import UserError
from .models import digest


@contextmanager
def directory_lock(folder: Path):
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise UserError("Эта папка результатов уже обрабатывается другим запуском.") from exc
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


class State:
    def __init__(self, path: Path):
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, source TEXT NOT NULL, target TEXT NOT NULL,
                fingerprint TEXT NOT NULL, settings TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending', output_hash TEXT, error TEXT
            );
            CREATE TABLE IF NOT EXISTS chunks (
                job_id TEXT NOT NULL REFERENCES jobs(id),
                core_start INTEGER NOT NULL, core_end INTEGER NOT NULL,
                text TEXT NOT NULL, speech_start REAL, speech_end REAL,
                PRIMARY KEY (job_id, core_start)
            );
        """)
        columns = {row[1] for row in self.db.execute("PRAGMA table_info(chunks)")}
        if "tail" not in columns:
            self.db.execute("ALTER TABLE chunks ADD COLUMN tail TEXT NOT NULL DEFAULT '[]'")
            self.db.commit()

    def job(self, source: Path, target: Path, fingerprint: str, settings: dict) -> str:
        metadata = json.dumps(settings, sort_keys=True)
        key = json.dumps([str(source), str(target), fingerprint, metadata])
        job_id = hashlib.sha256(key.encode()).hexdigest()
        with self.db:
            self.db.execute(
                "INSERT OR IGNORE INTO jobs(id,source,target,fingerprint,settings) "
                "VALUES (?,?,?,?,?)",
                (job_id, str(source), str(target), fingerprint, metadata),
            )
        return job_id

    def completed(self, job: str, target: Path) -> bool:
        status, output_hash = self.db.execute(
            "SELECT status,output_hash FROM jobs WHERE id=?",
            (job,),
        ).fetchone()
        matches = (
            status in {"complete", "exporting"}
            and target.is_file()
            and digest(target) == output_hash
        )
        if matches and status == "exporting":
            with self.db:
                self.db.execute("UPDATE jobs SET status='complete' WHERE id=?", (job,))
        return matches

    def reset(self, job: str):
        with self.db:
            self.db.execute("DELETE FROM chunks WHERE job_id=?", (job,))
            self.db.execute(
                "UPDATE jobs SET status='pending',output_hash=NULL,error=NULL WHERE id=?", (job,)
            )

    def status(self, job: str) -> str:
        return self.db.execute("SELECT status FROM jobs WHERE id=?", (job,)).fetchone()[0]

    def position(self, job: str) -> int:
        return self.db.execute(
            "SELECT COALESCE(MAX(core_end),0) FROM chunks WHERE job_id=?", (job,)
        ).fetchone()[0]

    def save(
        self,
        job: str,
        start: int,
        end: int,
        text: str,
        speech_start: float | None,
        speech_end: float | None,
        tail: list[dict] | None = None,
    ):
        if end <= start or start != self.position(job):
            raise UserError("Нарушена последовательность фрагментов в сохранённом прогрессе.")
        with self.db:
            self.db.execute(
                "INSERT INTO chunks VALUES(?,?,?,?,?,?,?)",
                (job, start, end, text, speech_start, speech_end, json.dumps(tail or [])),
            )
            self.db.execute("UPDATE jobs SET status='running',error=NULL WHERE id=?", (job,))

    def tail(self, job: str) -> list[dict]:
        row = self.db.execute(
            "SELECT tail FROM chunks WHERE job_id=? ORDER BY core_start DESC LIMIT 1",
            (job,),
        ).fetchone()
        return json.loads(row[0]) if row else []

    def fail(self, job: str, error: str):
        with self.db:
            self.db.execute(
                "UPDATE jobs SET status=CASE WHEN status='exporting' THEN status "
                "ELSE 'failed' END,error=? WHERE id=?",
                (error, job),
            )

    def export(self, job: str, target: Path, overwrite: bool):
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=".speech2text-", suffix=".tmp", dir=target.parent)
        temporary = Path(name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as out:
                first, previous_end, previous_text, paragraph_length = True, None, "", 0
                rows = self.db.execute(
                    "SELECT text,speech_start,speech_end FROM chunks "
                    "WHERE job_id=? ORDER BY core_start",
                    (job,),
                )
                for text, start, end in rows:
                    if not text:
                        continue
                    pause = (
                        start is not None
                        and previous_end is not None
                        and start - previous_end >= 1.5
                    )
                    paragraph = pause or (
                        paragraph_length >= 800 and previous_text.endswith((".", "!", "?"))
                    )
                    if not first:
                        out.write("\n\n" if paragraph else " ")
                    out.write(text)
                    paragraph_length = len(text) if paragraph else paragraph_length + len(text)
                    first, previous_end, previous_text = False, end, text
                if not first:
                    out.write("\n")
                out.flush()
                os.fsync(out.fileno())
            expected_hash = digest(temporary)
            with self.db:
                self.db.execute(
                    "UPDATE jobs SET status='exporting',output_hash=? WHERE id=?",
                    (expected_hash, job),
                )
            if overwrite:
                os.replace(temporary, target)
            else:
                # Atomic create-if-absent; another process cannot be silently overwritten.
                try:
                    os.link(temporary, target)
                except FileExistsError as exc:
                    raise UserError(
                        f"Результат уже существует: {target}. Нужен --overwrite."
                    ) from exc
            with self.db:
                self.db.execute(
                    "UPDATE jobs SET status='complete',output_hash=?,error=NULL WHERE id=?",
                    (expected_hash, job),
                )
        finally:
            temporary.unlink(missing_ok=True)

    def close(self):
        self.db.close()
