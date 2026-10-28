"""Saving to and loading from a SQLite file. No decisions are made here.

Two rules this file enforces, and the reason for each:

  1. Events are only ever inserted — never updated, never deleted. The event log is our record
     of the real conversation. Compaction rewrites what the *model* sees, but if we also edited
     the log we would have nothing left to restore from.

  2. Units are never deleted either. Re-splitting the conversation writes a whole new
     "generation" of units rather than replacing the old ones. `units()` gives you the newest
     generation and `unit(id)` finds a unit from any generation, so if the model is still
     holding a thread id from before a re-split, asking to reopen it still works.
"""

import hashlib
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from bonsai.models import Annotation, Event, Mark, Unit

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
  id TEXT PRIMARY KEY, session_id TEXT, seq INTEGER,
  role TEXT, type TEXT, content TEXT, tokens INTEGER, created_at TEXT);
CREATE TABLE IF NOT EXISTS units (
  id TEXT PRIMARY KEY, session_id TEXT, generation INTEGER, parent_id TEXT,
  label TEXT, synopsis TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS unit_events (unit_id TEXT, event_id TEXT);
CREATE TABLE IF NOT EXISTS annotations (
  id TEXT PRIMARY KEY, session_id TEXT, event_id TEXT, level REAL);
CREATE TABLE IF NOT EXISTS token_counts (hash TEXT PRIMARY KEY, tokens INTEGER);
CREATE TABLE IF NOT EXISTS marks (
  id TEXT PRIMARY KEY, session_id TEXT, event_id TEXT, kind TEXT, text TEXT,
  created_at TEXT);
"""

DEFAULT_PATH = Path.home() / ".bonsai" / "bonsai.db"


def _digest(text: str) -> str:
    """Hash rather than the text itself: trace strings are long and we only need identity."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.now(UTC).isoformat()


class Store:
    """One SQLite file holding every conversation bonsai has seen."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or DEFAULT_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        # Makes rows come back like dicts (row["label"]) instead of tuples (row[4]), which
        # keeps the queries below readable when columns get added.
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    # --- events ---------------------------------------------------------------

    def add_event(self, event: Event) -> None:
        """Append one message to the log."""
        with self.conn:
            self.conn.execute(
                "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    event.id,
                    event.session_id,
                    event.seq,
                    event.role,
                    event.type,
                    event.content,
                    event.tokens,
                    event.created_at,
                ),
            )

    def events(self, session_id: str) -> list[Event]:
        """Every message in one conversation, oldest first."""
        rows = self.conn.execute(
            "SELECT id, session_id, seq, role, type, content, tokens, created_at "
            "FROM events WHERE session_id = ? ORDER BY seq",
            (session_id,),
        ).fetchall()
        return [Event(**dict(row)) for row in rows]

    # --- units ----------------------------------------------------------------

    def replace_units(self, session_id: str, new_units: list[Unit]) -> list[Unit]:
        """Save a fresh grouping of the conversation and return it with its final ids.

        The store hands out the ids rather than letting the caller choose them, in the form
        `g<generation>u<unit>` for a unit and `g<generation>u<unit>.<thread>` for a thread —
        so `g3u2.1` is thread 1 of unit 2, from the third grouping. Whoever produced the
        grouping can number things however it likes; we renumber on the way in, so two
        groupings can never hand us colliding ids.

        A top-level unit gets its children's events rolled up into its own `event_ids`. That
        way "reopen this" works whether you point at a whole unit or one thread inside it.

        The generation number counts groupings across *every* conversation, not within one.
        That looks odd — a single conversation's generations come out as 1, 4, 7 rather than
        1, 2, 3 — but it is deliberate. Unit ids have to be unique across the whole database,
        because `unit(id)` looks one up without being told which conversation it came from. If
        each conversation counted its own generations, every conversation's first grouping
        would mint `g1u0` and collide on the primary key. The gaps cost nothing.
        """
        now = _now()
        with self.conn:
            generation = self.conn.execute(
                "SELECT COALESCE(MAX(generation), 0) + 1 FROM units"
            ).fetchone()[0]

            # Work out the final id for every unit before writing anything, so a thread can
            # look up the id its parent is about to get.
            parents = [unit for unit in new_units if unit.parent_id is None]
            final_id = {unit.id: f"g{generation}u{k}" for k, unit in enumerate(parents)}
            for parent in parents:
                children = [u for u in new_units if u.parent_id == parent.id]
                for j, child in enumerate(children):
                    final_id[child.id] = f"{final_id[parent.id]}.{j}"

            saved = []
            for unit in new_units:
                unit_id = final_id[unit.id]
                parent_id = final_id[unit.parent_id] if unit.parent_id else None
                event_ids = list(unit.event_ids)
                is_top_level = unit.parent_id is None
                if is_top_level and not event_ids:
                    event_ids = [
                        eid
                        for child in new_units
                        if child.parent_id == unit.id
                        for eid in child.event_ids
                    ]
                self.conn.execute(
                    "INSERT INTO units VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        unit_id,
                        unit.session_id,
                        generation,
                        parent_id,
                        unit.label,
                        unit.synopsis,
                        now,
                    ),
                )
                self.conn.executemany(
                    "INSERT INTO unit_events VALUES (?, ?)",
                    [(unit_id, eid) for eid in event_ids],
                )
                saved.append(
                    Unit(unit_id, unit.session_id, unit.label, event_ids, unit.synopsis, parent_id)
                )
        return saved

    def units(self, session_id: str) -> list[Unit]:
        """The newest grouping of one conversation, in the order it was written."""
        rows = self.conn.execute(
            "SELECT id, session_id, label, synopsis, parent_id FROM units WHERE session_id = ? "
            "AND generation = (SELECT MAX(generation) FROM units WHERE session_id = ?) "
            "ORDER BY rowid",
            (session_id, session_id),
        ).fetchall()
        return [Unit(**dict(row), event_ids=self._event_ids_for(row["id"])) for row in rows]

    def unit(self, unit_id: str) -> Unit | None:
        """One unit by id, from any grouping — this is what makes an old id still work."""
        row = self.conn.execute(
            "SELECT id, session_id, label, synopsis, parent_id FROM units WHERE id = ?",
            (unit_id,),
        ).fetchone()
        if row is None:
            return None
        return Unit(**dict(row), event_ids=self._event_ids_for(unit_id))

    def _event_ids_for(self, unit_id: str) -> list[str]:
        rows = self.conn.execute(
            "SELECT event_id FROM unit_events WHERE unit_id = ? ORDER BY rowid", (unit_id,)
        ).fetchall()
        return [row["event_id"] for row in rows]

    # --- annotations -------------------------------------

    def add_annotation(self, annotation: Annotation) -> None:
        """Record a pin / prune / level the user set on one message."""
        with self.conn:
            self.conn.execute(
                "INSERT INTO annotations VALUES (?, ?, ?, ?)",
                (
                    annotation.id,
                    annotation.session_id,
                    annotation.event_id,
                    annotation.level,
                ),
            )

    def annotations(self, session_id: str) -> list[Annotation]:
        rows = self.conn.execute(
            "SELECT id, session_id, event_id, level FROM annotations "
            "WHERE session_id = ? ORDER BY rowid",
            (session_id,),
        ).fetchall()
        return [Annotation(**dict(row)) for row in rows]

    def clear_annotations(self, session_id: str) -> int:
        """Forget every keep and drop set on one conversation. Returns how many went.

        The one place anything is deleted. Annotations are your choices rather than a record of
        what happened, so unlike events and threads they are yours to take back.
        """
        with self.conn:
            cur = self.conn.execute("DELETE FROM annotations WHERE session_id = ?", (session_id,))
        return cur.rowcount

    def add_mark(self, mark: Mark) -> None:
        """Record a note or a label you wrote."""
        with self.conn:
            self.conn.execute(
                "INSERT INTO marks VALUES (?, ?, ?, ?, ?, ?)",
                (mark.id, mark.session_id, mark.event_id, mark.kind, mark.text, mark.created_at),
            )

    def marks(self, session_id: str) -> list[Mark]:
        """Everything you wrote on one conversation, oldest first."""
        rows = self.conn.execute(
            "SELECT id, session_id, event_id, kind, text, created_at FROM marks "
            "WHERE session_id = ? ORDER BY rowid",
            (session_id,),
        ).fetchall()
        return [Mark(**dict(row)) for row in rows]

    def clear_marks(self, session_id: str, event_ids: list[str], kind: str) -> int:
        """Remove marks of one kind from these messages. Returns how many went."""
        if not event_ids:
            return 0
        holes = ",".join("?" * len(event_ids))
        with self.conn:
            cur = self.conn.execute(
                f"DELETE FROM marks WHERE session_id = ? AND kind = ? AND event_id IN ({holes})",
                (session_id, kind, *event_ids),
            )
        return cur.rowcount

    # --- cached token counts --------------------------------------------------

    def cached_count(self, text: str) -> int | None:
        """A token count we have measured before, or None.

        Worth a table of its own. Printing a plan measures one trace per thread, and on a long
        conversation that was hundreds of sequential API round trips every single time -- about
        a minute to look at a plan you might want to look at five times while steering. The
        counts never change unless the text does, so they are keyed by the text itself.
        """
        row = self.conn.execute(
            "SELECT tokens FROM token_counts WHERE hash = ?", (_digest(text),)
        ).fetchone()
        return row["tokens"] if row else None

    def put_count(self, text: str, tokens: int) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO token_counts VALUES (?, ?)", (_digest(text), tokens)
            )
