# core/history.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any

# 간단한 인메모리 딕셔너리를 사용하여 세션별 대화 기록을 저장합니다.
# { "session_id_1": [message1, message2], "session_id_2": [...] }
# 이 변수는 외부에서 직접 접근하지 않도록 _(언더스코어)로 시작합니다.
_chat_histories: Dict[str, List[Dict[str, Any]]] = {}


class BaseChatHistory(ABC):
    """
    대화 기록 관리를 위한 기본 추상 클래스입니다.
    이 클래스를 상속받아 메모리, Redis, MongoDB 등 다양한 저장소를 구현할 수 있습니다.
    """

    @abstractmethod
    def get_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """세션 ID에 해당하는 모든 메시지를 가져옵니다."""
        pass

    @abstractmethod
    def add_messages(self, session_id: str, messages: List[Dict[str, Any]]):
        """세션에 여러 개의 새 메시지를 추가합니다."""
        pass

    @abstractmethod
    def clear(self, session_id: str):
        """세션의 모든 메시지를 삭제합니다."""
        pass


class InMemoryChatHistory(BaseChatHistory):
    """
    인메모리 딕셔너리를 사용하여 대화 기록을 관리하는 클래스입니다.
    """

    def get_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """세션 ID에 해당하는 메시지 목록을 반환합니다. 없으면 빈 리스트를 반환합니다."""
        # .copy()를 사용하여 외부에서 원본 기록이 수정되는 것을 방지합니다.
        return _chat_histories.get(session_id, []).copy()

    def add_messages(self, session_id: str, messages: List[Dict[str, Any]]):
        """세션에 새 메시지들을 추가합니다. 세션이 없으면 새로 생성합니다."""
        if session_id not in _chat_histories:
            _chat_histories[session_id] = []
        _chat_histories[session_id].extend(messages)

    def clear(self, session_id: str):
        """세션 ID에 해당하는 대화 기록을 삭제합니다."""
        if session_id in _chat_histories:
            _chat_histories.pop(session_id)


# --- History Store Singleton ---
# 애플리케이션 전체에서 단 하나의 history_store 인스턴스를 사용하도록 설정합니다.
# 나중에 Redis, MongoDB 등으로 변경하려면 이 부분의 할당문만 수정하면 됩니다.
# 예: history_store = RedisChatHistory(redis_url="...")
history_store: BaseChatHistory = InMemoryChatHistory()


def get_history_store() -> BaseChatHistory:
    """
    현재 설정된 대화 기록 저장소 인스턴스를 반환합니다.
    이 함수를 통해 애플리케이션의 다른 부분에서 저장소에 접근합니다.
    """
    return history_store
