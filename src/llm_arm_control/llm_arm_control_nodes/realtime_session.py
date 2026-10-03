"""Pure state policy for a wake-gated realtime voice session."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ResponseRequest:
    session_id: str
    turn_serial: int
    instructions: str | None


@dataclass
class RealtimeSessionState:
    """Own session and single-flight response state without I/O side effects."""

    generation: int = 0
    connected: bool = False
    configured: bool = False
    connecting: bool = False
    active: bool = False
    audio_upload_enabled: bool = False
    session_id: str = ""
    turn_serial: int = 0
    item_turns: dict[str, int] = field(default_factory=dict)
    handled_input_items: set[str] = field(default_factory=set)
    ignored_turns: set[int] = field(default_factory=set)
    tool_count: int = 0
    tool_argument_errors: int = 0
    response_active: bool = False
    response_turn: int = -1
    response_id: str = ""
    response_cancelled: bool = False
    response_cancel_sent: bool = False
    pending_response: ResponseRequest | None = None
    response_started_at: float = 0.0
    response_transition_started_at: float = 0.0
    recover_active_session: bool = False
    last_activity: float = 0.0

    def start_connect(self) -> int | None:
        if self.connecting:
            return None
        self.connecting = True
        self.generation += 1
        return self.generation

    def accepts(self, generation: int) -> bool:
        return int(generation) == self.generation

    def begin_session(self, session_id: str, now: float) -> bool:
        if not self.configured or self.active:
            return False
        self.active = True
        self.audio_upload_enabled = False
        self.session_id = session_id
        self.turn_serial = 0
        self.item_turns.clear()
        self.handled_input_items.clear()
        self.ignored_turns.clear()
        self.tool_count = 0
        self.tool_argument_errors = 0
        self.reset_response()
        self.last_activity = now
        return True

    def record_speech(self, item_id: str, now: float) -> int:
        self.tool_count = 0
        self.tool_argument_errors = 0
        if item_id and item_id in self.item_turns:
            turn = self.item_turns[item_id]
        else:
            self.turn_serial += 1
            turn = self.turn_serial
            if item_id:
                self.item_turns[item_id] = turn
        self.last_activity = now
        return turn

    def record_transcript(self, item_id: str, now: float) -> tuple[int, bool]:
        duplicate = bool(item_id and item_id in self.handled_input_items)
        if duplicate:
            return self.item_turns.get(item_id, self.turn_serial), True
        if item_id and item_id in self.item_turns:
            turn = self.item_turns[item_id]
        else:
            self.tool_count = 0
            self.tool_argument_errors = 0
            self.turn_serial += 1
            turn = self.turn_serial
            if item_id:
                self.item_turns[item_id] = turn
        if item_id:
            self.handled_input_items.add(item_id)
        self.last_activity = now
        return turn, False

    def queue_response(
        self, turn_serial: int, instructions: str | None, now: float
    ) -> tuple[bool, ResponseRequest]:
        request = ResponseRequest(self.session_id, int(turn_serial), instructions)
        if self.response_active:
            self.pending_response = request
            return False, request
        self.response_active = True
        self.response_turn = int(turn_serial)
        self.response_id = ""
        self.response_cancelled = False
        self.response_cancel_sent = False
        self.response_started_at = now
        self.response_transition_started_at = now
        self.last_activity = now
        return True, request

    def request_cancel(self, now: float) -> tuple[int, str] | None:
        if not self.response_active:
            return None
        self.response_cancelled = True
        if not self.response_id:
            return self.response_turn, ""
        if self.response_cancel_sent:
            return None
        self.response_cancel_sent = True
        self.response_transition_started_at = now
        return self.response_turn, self.response_id

    def response_started(self, response_id: str) -> tuple[int, bool, int]:
        if not self.response_active or not response_id:
            return -1, False, -1
        self.response_id = response_id
        pending_turn = self.pending_response.turn_serial if self.pending_response else -1
        if not self.response_cancelled:
            self.response_transition_started_at = 0.0
        return self.response_turn, self.response_cancelled, pending_turn

    def response_event_turn(self, response_id: str) -> int | None:
        if (
            not self.response_active
            or not self.response_id
            or not response_id
            or response_id != self.response_id
        ):
            return None
        return self.response_turn

    def finish_response(self, response_id: str = "") -> ResponseRequest | None:
        if not self.response_active:
            return None
        if response_id and self.response_id and response_id != self.response_id:
            return None
        pending, self.pending_response = self.pending_response, None
        self.reset_response(keep_pending=True)
        if (
            pending is not None
            and pending.session_id == self.session_id
            and pending.turn_serial == self.turn_serial
        ):
            return pending
        return None

    def reset_response(self, keep_pending: bool = False) -> None:
        self.response_active = False
        self.response_turn = -1
        self.response_id = ""
        self.response_cancelled = False
        self.response_cancel_sent = False
        if not keep_pending:
            self.pending_response = None
        self.response_started_at = 0.0
        self.response_transition_started_at = 0.0

    def connection_lost(self, reconnect: bool) -> bool:
        preserve = bool(reconnect and self.active and self.session_id)
        self.connected = False
        self.configured = False
        self.connecting = False
        self.audio_upload_enabled = False
        self.item_turns.clear()
        self.handled_input_items.clear()
        self.ignored_turns.clear()
        self.reset_response()
        self.recover_active_session = preserve
        return preserve

    def connection_ready(self, now: float) -> bool:
        self.connected = True
        self.configured = True
        self.connecting = False
        recovering = self.recover_active_session and bool(self.session_id)
        self.recover_active_session = False
        if recovering:
            self.active = True
            self.audio_upload_enabled = True
            self.last_activity = now
        return recovering

    def timeout_phase(
        self,
        now: float,
        transition_timeout: float,
        response_timeout: float,
    ) -> str:
        if not self.response_active:
            return ""
        if (
            self.response_transition_started_at > 0.0
            and now - self.response_transition_started_at > transition_timeout
        ):
            return "cancel_wait" if self.response_cancel_sent else "create_wait"
        if self.response_started_at > 0.0 and now - self.response_started_at > response_timeout:
            return "response_max"
        return ""

    def end_session(self) -> str:
        old_session = self.session_id
        self.active = False
        self.audio_upload_enabled = False
        self.session_id = ""
        self.item_turns.clear()
        self.handled_input_items.clear()
        self.ignored_turns.clear()
        self.reset_response()
        self.recover_active_session = False
        return old_session
