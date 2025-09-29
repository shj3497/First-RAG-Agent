# core/tools/base.py
from abc import ABC, abstractmethod
from typing import Any, Dict


class Tool(ABC):
    """
    모든 도구(Tool)가 상속받아야 하는 추상 기반 클래스(Abstract Base Class)입니다.
    이 클래스는 모든 도구가 일관된 인터페이스를 갖도록 강제하는 '설계도' 역할을 합니다.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        도구의 고유한 이름입니다. LLM이 어떤 도구를 호출할지 식별하는 데 사용됩니다.
        예: "rag_search_tool"
        """
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """
        도구의 기능에 대한 상세한 설명입니다.
        LLM은 이 설명을 보고 사용자의 질문에 가장 적합한 도구가 무엇인지 판단합니다.
        예: "회사 내부 문서, 제품 정보, 가이드라인에 대한 질문에 답변할 때 사용합니다."
        """
        pass

    @abstractmethod
    async def execute(self, **kwargs: Any) -> Any:
        """
        도구의 실제 로직을 수행하는 비동기 메서드입니다.
        LLM이 전달한 인자(kwargs)를 받아 작업을 처리하고, 그 결과를 반환합니다.
        """
        pass

    def to_openai_format(self) -> Dict[str, Any]:
        """
        이 도구를 OpenAI의 Tool Calling 기능에서 사용할 수 있는 JSON 형식으로 변환합니다.
        (아직 세부 구현은 하지 않습니다.)
        """
        # TODO: 추후 OpenAI의 Tool Calling 명세에 맞게 함수 본문을 구현해야 합니다.
        #       도구에 필요한 파라미터(예: 'query')를 JSON 스키마로 정의하는 부분이 포함됩니다.
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        }
