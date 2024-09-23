"""
codes related to Agent
"""

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables.history import RunnableWithMessageHistory


class OpenAIChatAgent:
    """
    ChatBot based on OpenAI API
    """

    __store = {}
    __static_id = 0

    @classmethod
    def __get_session_history(cls, session_id: str) -> BaseChatMessageHistory:
        if session_id not in cls.__store:
            cls.__store[session_id] = InMemoryChatMessageHistory()
        return cls.__store[session_id]

    def __init__(self, system_prompt: str = None):
        model = ChatOpenAI(model="gpt-4o-mini")
        if system_prompt:
            prompt = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(system_prompt),
                    MessagesPlaceholder(variable_name="messages"),
                ]
            )
        else:
            prompt = ChatPromptTemplate.from_messages(
                [MessagesPlaceholder(variable_name="messages")]
            )
        parser = StrOutputParser()
        chain = prompt | model | parser
        with_message_history = RunnableWithMessageHistory(
            chain, self.__get_session_history
        )
        session_id = self.__static_id
        self.__static_id += 1
        self.config = {"configurable": {"session_id": session_id}}
        self.chain = with_message_history | parser

    def clear_history(self) -> None:
        """
        Clear the history of current chat session
        """
        self.config["configurable"]["session_id"] = self.__static_id
        self.__static_id += 1

    def store_human(self, message: str) -> None:
        """store a human message without invoke chain immediately"""
        session_id: str = self.config["configurable"]["session_id"]
        history = self.__get_session_history(session_id)
        history.add_user_message(message)

    def store_ai(self, message: str) -> None:
        """store an ai message without invoke chain immediately"""
        session_id: str = self.config["configurable"]["session_id"]
        history = self.__get_session_history(session_id)
        history.add_ai_message(message)

    def post_human(self, message: str) -> str:
        """post a humane message"""
        return self.chain.invoke(
            {"messages": HumanMessage(message)}, config=self.config
        )

    def post_ai(self, message: str) -> str:
        """post a ai message"""
        return self.chain.invoke({"messages": AIMessage(message)}, config=self.config)

    def post(self) -> str:
        """post based on current history"""
        return self.chain.invoke({"messages": []}, config=self.config)
