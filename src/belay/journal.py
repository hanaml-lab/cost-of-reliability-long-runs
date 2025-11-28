"""A minimal durable executor (Section 5 of the report).

A job is a list of steps. Each step's result is appended to a journal before the next step runs. On
restart, ``run`` replays the journal: recorded results are returned without executing the step, so
model calls are not re-sampled and the agent's past decisions do not change. A step that sends an
external side effect carries an idempotency key made from the job and the step; the service stores
the key with its result and answers a repeated key with the stored result, which closes the window in
which the effect was applied but the worker died before journalling it.

This is the protocol the report's runtime implements on a workflow engine with a PostgreSQL journal;
here it is in memory so that its guarantees can be tested directly.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable


class Crash(Exception):
    """The worker died."""


@dataclass(frozen=True)
class Step:
    name: str
    kind: str  # "model", "tool" or "effect"
    fn: Callable[[list], Any]


@dataclass
class Journal:
    events: list = field(default_factory=list)


@dataclass
class Service:
    """An external system. With a key, a repeated request returns the stored result."""

    log: list = field(default_factory=list)
    stored: dict = field(default_factory=dict)

    def call(self, step: str, payload: Any, key: str | None) -> str:
        if key is not None and key in self.stored:
            return self.stored[key]
        result = f"{step}:ok:{len(self.log)}"
        self.log.append((step, repr(payload)))
        if key is not None:
            self.stored[key] = result
        return result

    def executed(self, step: str) -> int:
        return sum(1 for s, _ in self.log if s == step)

    def duplicates(self) -> int:
        return sum(c - 1 for c in Counter(self.log).values())


def run(steps: list[Step], journal: Journal, service: Service, *, job: str, idempotency: bool = True, crash: tuple[str, str] | None = None) -> list:
    """Run (or resume) a job. ``crash=(step, "before" | "in-flight")`` kills the worker at that step:
    before it starts, or after its side effect was sent and before its result was journalled."""
    state = list(journal.events)
    for i, step in enumerate(steps):
        if i < len(journal.events):
            continue  # replay: the recorded result stands
        if crash == (step.name, "before"):
            raise Crash(step.name)
        out = step.fn(state)
        if step.kind == "effect":
            key = f"{job}/{i}" if idempotency else None
            out = service.call(step.name, out, key)
            if crash == (step.name, "in-flight"):
                raise Crash(step.name)
        journal.events.append(out)
        state.append(out)
    return state
