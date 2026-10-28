---
description: Run any bonsai command on this conversation
argument-hint: "[status | threads -v | plan --budget N | keep \"...\" | note ID \"...\" | ...]"
allowed-tools: Bash(bonsai *), PowerShell(bonsai *)
---

Run bonsai with whatever I passed, on this conversation.

    bonsai $ARGUMENTS

If `$ARGUMENTS` is empty, run `bonsai status` and then `bonsai threads -v`.

**On Windows, `bonsai` may not be on the Bash path.** If Bash reports `command not found`, run
the same command through PowerShell instead. Do not report failure until you have tried both.

**Paste the output verbatim, in a code block.** The ids, turn spans, token counts and labels are
the entire point — do not summarize them away, and do not leave them only inside a collapsed
tool result that I would have to click to open.

Then add at most two or three sentences of your own:

- If `status` shows `NOT INDEXED`, say so and give the number of turns. Those turns are in no
  thread until `bonsai segment` runs.
- If I ran `plan`, say which threads stop being verbatim and whether any hold something the
  current work depends on. Name them by id.
- If the command rewrote the transcript (`compact`), finish with this as the last line and
  nothing after it:

  **Reload now to pick up the compacted transcript: Ctrl+Shift+P → "Developer: Reload Window".**

Two things not to do on your own initiative: `bonsai segment` is one model call over the whole
transcript — minutes and real money on a long session — and `bonsai compact` rewrites my
session file. Run either only if I actually asked for it in `$ARGUMENTS`.

The commands available are: `status`, `threads`, `plan`, `compact`, `segment`, `keep`, `drop`,
`focus`, `steering`, `recall`, `note`, `relabel`, `merge`, `split`.
