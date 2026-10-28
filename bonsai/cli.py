"""The `bonsai` command. Works on Claude Code sessions.

    bonsai threads [-v]                  the conversation as units and threads
    bonsai segment [--by "..."] [--units N] [--subunits M]   re-split it, your way
    bonsai plan --budget N               what a compaction would keep, and why
    bonsai compact --budget N            show the plan, then rewrite the transcript
    bonsai keep "<description>" | ID     keep matching threads verbatim
    bonsai drop "<description>" | ID     collapse matching threads to their label
    bonsai focus "<description>"         keep what matches, collapse the rest
    bonsai steering [--clear]            what you have kept and dropped so far
    bonsai recall ID                     a collapsed thread's original text
    bonsai note ID "..."                 attach your own note; it survives compaction
    bonsai relabel ID "..."              replace the label with your own wording
    bonsai merge ID ID [ID...]           join threads the segmenter split apart
    bonsai split ID TURN                 cut a thread in two at a turn number
    bonsai status                        where this conversation stands right now

Every command takes an optional SESSION path (a Claude Code .jsonl). Leave it off and it uses
the newest session for whatever directory you're in. A compaction takes effect the next time
the session is loaded: in VS Code, run "Developer: Reload Window". `--model` picks the model
for every API call.

The functions here return text instead of printing it, so other front ends can show the same
views without duplicating the formatting.
"""

import argparse
import os
import re
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from bonsai import compact, llm, transcript
from bonsai.models import Unit
from bonsai.session import TAIL_ID, Session

PROJECTS = Path.home() / ".claude" / "projects"


def load_env(path: Path | None = None) -> None:
    """Read `KEY=value` lines from a local .env into the environment, if one is there.

    `.gitignore` has listed `.env` since the first commit, which implies bonsai reads one. It
    did not, so every command needed `ANTHROPIC_API_KEY` exported by hand first — easy to
    forget, and a clumsy thing to put in a README.

    Anything already set in the shell wins: that is the more deliberate choice of the two, and
    silently overriding it would be a nasty surprise. Written out rather than taking a
    dependency, because it is a dozen lines and bonsai has exactly one dependency today.
    """
    env = path or Path.cwd() / ".env"
    if not env.is_file():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def project_folder(path: str) -> str:
    """The folder name Claude Code uses for a working directory, lowercased for matching.

    Every character that is not a letter or a digit becomes a dash — not only the separators.
    `C:\\Vault\\Justin's Vault (Sync)` becomes `C--Vault-Justin-s-Vault--Sync-`: the apostrophe,
    both spaces and both brackets each turn into one dash.

    An earlier version replaced only `:` `\\` and `/`, so any path containing a space or any
    punctuation could not be matched at all — and the error blamed a missing session rather
    than the name bonsai had built.
    """
    return re.sub(r"[^a-zA-Z0-9]", "-", path).lower()


def find_session(arg: str | None = None) -> Path:
    """The .jsonl you named, or the newest Claude Code session for the current directory."""
    if arg:
        return Path(arg)
    folder_name = project_folder(str(Path.cwd()))
    folders = [d for d in PROJECTS.iterdir() if d.name.lower() == folder_name]
    files = [f for d in folders for f in d.glob("*.jsonl") if ".bak" not in f.name]
    if not files:
        # Name the folder we looked for, not just the directory we are in: when this fails it
        # is almost always a name mismatch, and showing both makes that obvious at a glance.
        raise SystemExit(
            f"no Claude Code sessions found for {Path.cwd()}\n"
            f"  looked for: {PROJECTS / folder_name}\n"
            "  if that folder has a different name, pass the .jsonl path directly:\n"
            "      bonsai status <path-to-session.jsonl>"
        )
    return max(files, key=lambda f: f.stat().st_mtime)


SEGMENT_FIRST = (
    "This conversation has not been split into threads yet, so there is nothing to plan over.\n"
    "Splitting is one model call over the whole transcript: minutes of waiting and real money\n"
    "on a long session, which is why nothing starts it for you. Run it yourself:\n"
    "    bonsai segment\n"
    "Everything after that is instant and free; the threads are cached."
)


def segmented(session: Session) -> bool:
    """True when every message already belongs to a thread, so nothing will trigger a model call.

    Worth checking before doing anything that's supposed to be fast: the first call on an
    unsplit conversation quietly costs minutes and real money.
    """
    units = session.store.units(session.session_id)
    covered = {eid for unit in units for eid in unit.event_ids}
    return bool(units) and all(event.id in covered for event in session.events())


def load(path: Path, model: str = llm.MODEL) -> tuple[Session, list[transcript.Line]]:
    """Open a Claude Code session file and make sure every message is in the event log."""
    lines = transcript.read(path)
    session = Session.open(transcript.session_id(lines), model=model)
    transcript.import_events(session, lines)
    return session, lines


def threads(
    session: Session,
    lines: list[transcript.Line],
    verbose: bool = False,
    recent: bool = False,
) -> str:
    """The conversation as units and threads: size, state, your settings, label.

    Ordered by where each topic sits in the conversation, which is how you remember it. With
    `recent`, newest topics come first instead -- the right order when you are picking up work
    after a break and only the last few matter.
    """
    events = {event.id: event for event in session.events()}
    units = session.units()
    state = transcript.collapsed(lines)
    fixed = compact.overrides(units, session.store.annotations(session.session_id))

    def last_turn(unit: Unit) -> int:
        return max((events[eid].seq for eid in unit.event_ids), default=-1)

    if recent:
        shown = [(u, None) for u in sorted(units, key=last_turn, reverse=True) if u.parent_id]
    else:
        shown = [(u, None) for u in units]

    out = []
    for unit, _ in shown:
        tokens = sum(events[eid].tokens for eid in unit.event_ids)
        level = fixed.get(unit.id)
        knob = "" if level is None else "keep" if level >= 1 else "drop"
        indent = "" if recent or not unit.parent_id else "    "
        span = compact.span_text(unit, events)
        # The indent goes *inside* the id field rather than before it, so a child row's later
        # columns land in the same place as a parent's instead of being pushed four to the right.
        out.append(
            f"{indent + unit.id:13} {tokens:7,} tok {state.get(unit.id, 'full'):5} "
            f"{knob:4} {span:20} {unit.label}"
        )
        if unit.note:
            out.append(f"{'':26}note: {unit.note}")
        if verbose and unit.synopsis:
            out.append(f"{'':26}{unit.synopsis}")

    unit_count = sum(1 for unit in units if not unit.parent_id)
    total = sum(event.tokens for event in events.values())
    out.append(f"\n{len(compact.leaves(units))} threads in {unit_count} units, {total:,} tokens")
    return "\n".join(out)


def status(session: Session, lines: list[transcript.Line]) -> str:
    """Where this conversation stands right now.

    `threads` says what is in the conversation and `plan` says what you would get if you
    compacted. Neither answers "what is true at this moment", which is what you want after
    coming back to a session. The stale-index line matters most: turns said since the last
    split are not in any thread, and nothing else tells you that.
    """
    events = session.events()
    units = session.units()
    # leaves(), not "has a parent": the stand-in thread holding turns said since the last
    # split is a top-level unit with no parent, and filtering on parent_id hid it completely --
    # which meant the stale-index warning below could never fire.
    all_threads = compact.leaves(units)
    state = transcript.collapsed(lines)
    steered = session.steering()
    visible = [ln for ln in lines if ln.is_conversation]
    fresh = [u for u in all_threads if u.id == TAIL_ID]
    collapsed_now = sum(1 for u in all_threads if state.get(u.id) == "trace")

    rows = [
        f"transcript   {len(lines):,} lines - {len(visible):,} messages reach the model",
        f"logged       {len(events):,} messages, {sum(e.tokens for e in events):,} tokens",
        (
            f"threads      {len(all_threads)} in {sum(1 for u in units if not u.parent_id)} units"
            f" - {collapsed_now} collapsed in the file"
        ),
        (
            f"steered      {sum(1 for v in steered.values() if v == 'keep')} kept,"
            f" {sum(1 for v in steered.values() if v == 'drop')} dropped"
        ),
        f"notes        {sum(1 for u in all_threads if u.note)} thread(s) carry one",
        "mode         "
        + (
            "READ-ONLY (BONSAI_READONLY is set; compaction refuses)"
            if transcript.read_only()
            else "normal (compaction may rewrite this transcript)"
        ),
    ]
    if fresh:
        rows.append(
            f"NOT INDEXED  {len(fresh[0].event_ids)} turns since the last split are in no thread"
            " - run `bonsai segment` to fold them in"
        )
    return "\n".join(rows)


def plan(session: Session, budget: int) -> tuple[compact.Plan, str]:
    """The plan and the text showing it. Exits with advice if the budget can't be met."""
    if not segmented(session):
        raise SystemExit(SEGMENT_FIRST)
    try:
        the_plan = session.plan(budget)
    except compact.BudgetError as e:
        raise SystemExit(f"{e}\n  drop something you kept, or raise --budget") from None

    events = {event.id: event for event in session.events()}
    out = []
    for unit in session.units():
        if unit.id not in the_plan.detail:
            out.append(f"{unit.id:9} {'':9} {unit.label}")  # a parent unit, just a heading
            continue
        tokens = sum(events[eid].tokens for eid in unit.event_ids)
        why = the_plan.reasons.get(unit.id, "")
        out.append(
            f"    {unit.id:9} {the_plan.detail[unit.id]:6} {tokens:7,} tok  {why:10} {unit.label}"
        )

    counts = {
        kind: sum(1 for v in the_plan.detail.values() if v == kind) for kind in ("full", "trace")
    }
    out.append(
        f"\n{counts['full']} kept in full, {counts['trace']} collapsed"
        f"  ->  ~{the_plan.estimate:,} tokens"
    )
    if counts["trace"]:
        out.append(
            f"{counts['trace']} thread(s) are trace only at this budget: label and synopsis, with "
            "the original one `recall` away. Raise --budget or `keep` what you need verbatim."
        )
    return the_plan, "\n".join(out)


def apply(
    session: Session, lines: list[transcript.Line], the_plan: compact.Plan, path: Path
) -> str:
    """Back up the transcript, then rewrite it according to the plan.

    The ordering here matters. Generating summaries takes minutes, and if the session is live
    Claude Code is appending to the file the whole time — including the messages in which you
    asked for the compaction. So after the rewrite is built we re-read the file, find the lines
    that arrived while we were working, and carry them onto the end with their parent pointer
    repaired.

    Without that step those lines survive but point at a message the rewrite replaced, which
    breaks the chain and makes the entire conversation before them unreachable.
    """
    # Checked before any work, not just before the write: the backup below also lands outside
    # ~/.bonsai/, and doing minutes of work to then refuse would be its own kind of rude.
    if transcript.read_only():
        raise SystemExit(
            "BONSAI_READONLY is set, so nothing was written. Planning, steering, notes and "
            "recall all still work; unset it to allow compaction."
        )
    started = time.time()
    new_lines = transcript.rewrite(lines, session, session.units(), the_plan)

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    backup = path.with_suffix(f".{stamp}.bak.jsonl")
    shutil.copy2(path, backup)

    fresh = transcript.read(path)
    seen = {line.uuid for line in lines if line.uuid}
    arrived = [line for line in fresh[len(lines) :] if line.uuid not in seen]
    new_lines += transcript.rechain(arrived, after=new_lines)
    transcript.write(path, new_lines)

    minutes, seconds = divmod(int(time.time() - started), 60)
    note = f"\n{len(arrived)} line(s) arrived while compacting and were kept." if arrived else ""
    return (
        f"\n=== COMPACTION DONE in {minutes}m{seconds:02d}s ==="
        f"\n{path.name}: {len(lines)} -> {len(new_lines)} lines"
        f"\nbackup: {backup.name}{note}"
        "\n\nNOTHING HAS CHANGED IN YOUR SESSION YET. Reload to load it:"
        "\n    Ctrl+Shift+P  ->  Developer: Reload Window"
    )


def set_level(session: Session, unit_id: str, level: float) -> str:
    """Record a choice on every message in a thread.

    On the messages rather than the thread, so it survives a later re-split. Re-segmenting
    regroups threads, and anything attached to a thread id would quietly vanish.
    """
    unit = session.store.unit(unit_id)
    if unit is None:
        raise SystemExit(f"no thread {unit_id}")
    for event_id in unit.event_ids:
        session.level(event_id, level)
    return f"{'keep' if level >= 1 else 'drop'}: {unit.id}  {unit.label}"


def steer(session: Session, target: str, level: float, model: str) -> str:
    """Apply a choice to threads named by id, or found by description.

    An id is used directly. Anything else is treated as a description and matched against the
    labels and synopses, because reading 188 labels to find three ids is not something anyone
    will actually do -- and steering nobody uses is the same as no steering.

    If the match fails for any reason, nothing is changed and the error says so. Silently
    pinning the wrong threads would be worse than doing nothing.
    """
    if session.store.unit(target) is not None:
        return set_level(session, target, level)

    threads = [u for u in session.units() if u.parent_id]
    if not threads:
        raise SystemExit("this conversation has no threads yet; run `bonsai threads` first")
    try:
        chosen = llm.match_threads(target, threads, model)
    except Exception as e:  # noqa: BLE001 - any failure here must leave the session untouched
        raise SystemExit(
            f"could not match {target!r}: {type(e).__name__}: {e}\n  nothing changed"
        ) from None
    if not chosen:
        return f"nothing matched {target!r}; nothing changed"
    return "\n".join(set_level(session, unit_id, level) for unit_id in chosen)


def steering(session: Session, clear: bool = False) -> str:
    """What you have kept and dropped so far, or clear it."""
    if clear:
        gone = session.clear_steering()
        return f"cleared {gone} setting(s); nothing is kept or dropped any more"
    state = session.steering()
    if not state:
        return "nothing kept or dropped yet"
    labels = {u.id: u.label for u in session.units()}
    rows = [f"{verb:5} {tid:9} {labels.get(tid, '')}" for tid, verb in sorted(state.items())]
    joined = "\n".join(rows)
    return f"{joined}\n\n{len(state)} thread(s) steered"


def focus(session: Session, target: str, model: str) -> str:
    """Keep what matches, collapse everything you have not already asked to keep.

    The common case when you switch tasks: this is what matters now, the rest is noise. It adds
    to your steering rather than replacing it -- threads you kept earlier stay kept, because
    quietly undoing a choice you made on purpose is worse than keeping something you did not
    need.
    """
    before = session.steering()
    kept_already = {tid for tid, verb in before.items() if verb == "keep"}
    out = [steer(session, target, 1.0, model)]

    now_kept = {tid for tid, verb in session.steering().items() if verb == "keep"}
    if now_kept == kept_already:
        return out[0]  # nothing matched; do not drop the whole conversation on a typo

    dropped = 0
    for thread in session.units():
        if thread.parent_id and thread.id not in now_kept:
            set_level(session, thread.id, 0.0)
            dropped += 1
    out.append(f"\ndropped {dropped} other thread(s); {len(now_kept)} kept")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> None:
    load_env()  # a .env beside you, so a fresh clone needs one file and no shell ceremony
    parser = argparse.ArgumentParser(
        prog="bonsai", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", default=llm.MODEL)
    sub = parser.add_subparsers(dest="cmd", required=True)

    threads_cmd = sub.add_parser("threads")
    threads_cmd.add_argument("-v", "--verbose", action="store_true")
    threads_cmd.add_argument("--recent", action="store_true", help="newest topics first")

    segment_cmd = sub.add_parser("segment")
    segment_cmd.add_argument("--by")
    segment_cmd.add_argument("--units", type=int, default=6)
    segment_cmd.add_argument("--subunits", default="3-10")

    plan_cmd = sub.add_parser("plan")
    compact_cmd = sub.add_parser("compact")
    for cmd in (plan_cmd, compact_cmd):
        cmd.add_argument("--budget", type=int, required=True)

    # keep, drop and focus take either a thread id or a description; recall takes an id.
    for name in ("keep", "drop", "focus", "recall"):
        sub.choices[name] = sub.add_parser(name)
        sub.choices[name].add_argument("target")

    steering_cmd = sub.add_parser("steering")
    steering_cmd.add_argument("--clear", action="store_true")

    sub.add_parser("status")

    for name in ("note", "relabel"):
        sub.choices[name] = sub.add_parser(name)
        sub.choices[name].add_argument("unit_id")
        sub.choices[name].add_argument("text")

    merge_cmd = sub.add_parser("merge")
    merge_cmd.add_argument("unit_ids", nargs="+")
    merge_cmd.add_argument("--label")

    split_cmd = sub.add_parser("split")
    split_cmd.add_argument("unit_id")
    split_cmd.add_argument("at_turn", type=int)

    for cmd in sub.choices.values():
        cmd.add_argument("session", nargs="?")
    args = parser.parse_args(argv)

    path = find_session(args.session)
    session, lines = load(path, args.model)

    if args.cmd == "threads":
        print(threads(session, lines, args.verbose, args.recent))
    elif args.cmd == "segment":
        session.segment(args.by, args.units, args.subunits)
        print(threads(session, lines))
    elif args.cmd == "plan":
        print(plan(session, args.budget)[1])
    elif args.cmd == "compact":
        the_plan, text = plan(session, args.budget)
        # The plan is printed rather than withheld behind a prompt: `bonsai plan` is the step
        # for looking, and asking again about something this reversible mostly teaches people
        # to confirm without reading. The transcript is backed up before the rewrite, the event
        # log is untouched, and compacting again at a larger budget brings the text back.
        print(text, flush=True)
        print(apply(session, lines, the_plan, path))
    elif args.cmd == "recall":
        print(session.recall(args.target))
    elif args.cmd == "steering":
        print(steering(session, args.clear))
    elif args.cmd == "focus":
        print(focus(session, args.target, args.model))
    elif args.cmd == "status":
        print(status(session, lines))
    elif args.cmd == "note":
        print(f"noted on {session.note(args.unit_id, args.text).id}")
    elif args.cmd == "relabel":
        print(f"relabelled {session.relabel(args.unit_id, args.text).id}: {args.text}")
    elif args.cmd == "merge":
        session.merge(args.unit_ids, args.label)
        print(threads(session, lines))
    elif args.cmd == "split":
        session.split(args.unit_id, args.at_turn)
        print(threads(session, lines))
    else:
        print(steer(session, args.target, 1.0 if args.cmd == "keep" else 0.0, args.model))


if __name__ == "__main__":
    main(sys.argv[1:])
