from collections import OrderedDict


class SessionStore:
    def __init__(self, factory, max_sessions=16):
        if max_sessions < 1:
            raise ValueError("max_sessions 必须大于等于 1")
        self.factory = factory
        self.max_sessions = max_sessions
        self._sessions = OrderedDict()

    def get(self, session_id):
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id 必须是非空字符串")
        if len(session_id) > 128:
            raise ValueError("session_id 不能超过 128 个字符")

        if session_id in self._sessions:
            session = self._sessions[session_id]
            self._sessions.move_to_end(session_id)
            return session

        session = self.factory()
        self._sessions[session_id] = session

        if len(self._sessions) > self.max_sessions:
            self._sessions.popitem(last=False)

        return session

    def find(self, session_id):
        return self._sessions.get(session_id)

    def list_sessions(self):
        result = []
        for session_id, session in reversed(self._sessions.items()):
            result.append({
                "session_id": session_id,
                "title": (
                    session.history[0][0]
                    if session.history
                    else "新会话"
                ),
                "turns": len(session.history),
            })
        return result

    def clear(self, session_id):
        self._sessions.pop(session_id, None)

    def __len__(self):
        return len(self._sessions)
