# core/base_agent.py
# -*- coding: utf-8 -*-
from typing import List
from openai import AsyncOpenAI
from langchain_openai import ChatOpenAI

from core.tools.base import Tool
from core.tools.rag_search import RagSearchTool

# --- 공통 설정 ---

# 사용할 OpenAI 모델을 지정합니다.
AGENT_MODEL = "gpt-4-turbo"

# 이 에이전트가 사용할 수 있는 도구들의 리스트입니다.
AVAILABLE_TOOLS: List[Tool] = [
    RagSearchTool(),
]

# --- 클라이언트 인스턴스 ---

# OpenAI 비동기 클라이언트는 지연 초기화합니다. (기존 agent.py 용)
_openai_client: AsyncOpenAI | None = None


def get_openai_client() -> AsyncOpenAI:
    """OpenAI의 비동기 클라이언트 인스턴스를 반환합니다."""
    global _openai_client
    if _openai_client is None:
        _openai_client = AsyncOpenAI()
    return _openai_client


def get_chat_model() -> ChatOpenAI:
    """LangChain의 ChatOpenAI 모델 인스턴스를 반환합니다."""
    return ChatOpenAI(model=AGENT_MODEL, temperature=0)
