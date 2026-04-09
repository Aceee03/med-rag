from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter


def format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "?:??:??"

    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


@dataclass
class ProgressPrinter:
    label: str
    total: int
    every: int = 25
    start_count: int = 0
    started_at: float = field(default_factory=perf_counter)
    last_printed_count: int = field(init=False)

    def __post_init__(self) -> None:
        self.every = max(1, int(self.every))
        self.last_printed_count = self.start_count

    def update(self, current_count: int, extra: str | None = None, *, force: bool = False) -> None:
        if self.total <= 0:
            return

        should_print = force
        should_print = should_print or current_count == 1
        should_print = should_print or current_count >= self.total
        should_print = should_print or (current_count - self.last_printed_count) >= self.every
        if not should_print:
            return

        self.last_printed_count = current_count
        processed_since_start = max(0, current_count - self.start_count)
        elapsed = perf_counter() - self.started_at
        rate = (processed_since_start / elapsed) if elapsed > 0 and processed_since_start > 0 else 0.0
        remaining = max(0, self.total - current_count)
        eta = (remaining / rate) if rate > 0 else None
        percent = (current_count / self.total) * 100

        message = (
            f"[{self.label}] {current_count}/{self.total} ({percent:5.1f}%)"
            f" | elapsed {format_duration(elapsed)}"
        )
        if rate > 0:
            message += f" | rate {rate:.2f}/s"
        if eta is not None:
            message += f" | eta {format_duration(eta)}"
        if extra:
            message += f" | {extra}"

        print(message, flush=True)
