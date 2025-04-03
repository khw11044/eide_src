
from typing import Any, AsyncIterable, Dict, Literal, Optional, Union

from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.prompts import MessagesPlaceholder
from langchain_community.callbacks import get_openai_callback
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_openai import AzureChatOpenAI, ChatOpenAI

from .prompts import RobotSystemPrompts, system_prompts
from .tools import ROSATools

ChatModel = Union[ChatOpenAI, AzureChatOpenAI, ChatOllama]


class ROSA:
    """
    ROSA (Robot Operating System Agent)는 자연어를 사용하여 ROS 시스템과 상호작용하는 로직을 캡슐화하는 클래스

    Note:
        - `tools` 및 `tool_packages` 인자를 사용하여 에이전트의 기능을 확장 가능
        - 특정 로봇 또는 사용 목적에 맞게 에이전트의 동작을 조정하기 위해 맞춤형 `prompts`를 인자로 제공 가능
        - 스트리밍이 활성화되면 토큰 사용량 표시 기능이 자동으로 비활성화
        - 스트리밍이 필요하지 않을 경우 `invoke()`를, 스트리밍이 필요할 경우 `astream()`을 사용

    """

    def __init__(
        self,
        llm: ChatModel,                                                     # 응답을 생성하는 데 사용할 언어 모델입니다.
        tools: Optional[list] = None,                                       # 에이전트와 함께 사용할 추가 LangChain 도구 함수 목록입니다.
        tool_packages: Optional[list] = None,                               # 에이전트와 함께 사용할 LangChain 도구 함수가 포함된 Python 패키지 목록입니다.
        prompts: Optional[RobotSystemPrompts] = None,                       # 에이전트에서 사용할 맞춤형 프롬프트입니다.
        verbose: bool = False,                                              # 자세한 출력을 표시할지 여부를 결정합니다. 기본값은 False입니다.
        blacklist: Optional[list] = None,                                   # 에이전트에서 제외할 ROS 도구 목록입니다.
        accumulate_chat_history: bool = True,                               # 채팅 기록을 누적할지 여부를 결정합니다. 기본값은 True입니다.
        show_token_usage: bool = False,                                     # 토큰 사용량을 표시할지 여부를 결정합니다. 스트리밍이 활성화되면 작동하지 않습니다. 기본값은 False입니다.
        streaming: bool = True,                                             # 에이전트의 출력을 스트리밍할지 여부를 결정합니다. 기본값은 True입니다.
    ):
        self.__chat_history = []                                            # 채팅 기록을 나타내는 메시지 목록입니다.
        self.__llm = llm.with_config({"streaming": streaming})
        self.__memory_key = "chat_history"
        self.__scratchpad = "agent_scratchpad"
        self.__blacklist = blacklist if blacklist else []
        self.__accumulate_chat_history = accumulate_chat_history
        self.__streaming = streaming
        self.__tools = self._get_tools(
            packages=tool_packages, tools=tools, blacklist=self.__blacklist
        )
        self.__prompts = self._get_prompts(prompts)
        self.__llm_with_tools = self.__llm.bind_tools(self.__tools.get_tools())
        self.__agent = self._get_agent()
        self.__executor = self._get_executor(verbose=verbose)
        self.__show_token_usage = show_token_usage if not streaming else False

    @property
    def chat_history(self):
        """채팅 기록 (리스트) 얻기."""
        return self.__chat_history

    def clear_chat(self):               
        """채팅 기록을 초기화"""
        self.__chat_history = []

    def invoke(self, query: str) -> str:            # query (str): 에이전트가 처리할 사용자 입력 질의.
        """
        이 메서드는 사용자의 질의를 에이전트를 통해 처리하고 응답을 반환하며, 토큰 사용량을 추적하고 채팅 기록을 업데이트
        
        Note:
            - get_openai_callback을 통해 토큰 사용량을 추적할 수 있음(활성화된 경우).
            - 질의와 응답이 성공적으로 처리되면 chat_history이 업데이트
        """
        
        try:
            with get_openai_callback() as cb:
                result = self.__executor.invoke(
                    {"input": query, "chat_history": self.__chat_history}
                )
                self._print_usage(cb)
        except Exception as e:
            return f"An error occurred: {str(e)}"

        self._record_chat_history(query, result["output"])
        return result["output"]


    async def astream(self, query: str) -> AsyncIterable[Dict[str, Any]]:       # query (str): 사용자의 입력 질의.
        """
        사용자의 질의에 대한 에이전트의 응답을 비동기적으로 스트리밍

        이 메서드는 사용자의 질의를 처리하고, 토큰 생성, 도구 사용, 최종 출력 등의 이벤트를 발생하는 즉시 전달
        스트리밍이 활성화된 경우에 사용하도록 설계되어 있음


        Returns:
            AsyncIterable[Dict[str, Any]]: 이벤트 정보를 포함하는 비동기 반복 가능한 딕셔너리 객체.
            각 딕셔너리는 'type' 키를 가지며, 이벤트 유형에 따라 추가 키가 포함됩니다.
            - 'token': 생성된 토큰을 'content'와 함께 반환.
            - 'tool_start': 도구 실행이 시작될 때 'name' 및 'input' 포함.
            - 'tool_end': 도구 실행이 종료될 때 'name' 및 'output' 포함.
            - 'final': 에이전트의 최종 출력을 'content'와 함께 제공.
            - 'error': 오류 발생 시 'content'에 오류 내용을 포함.

        Raises:
            ValueError: 이 ROSA 인스턴스에서 스트리밍이 활성화되지 않은 경우 발생.
            Exception: 스트리밍 과정에서 오류가 발생하면 예외를 발생.

        Note:
            - 이 메서드는 최종 출력이 성공적으로 생성되면 채팅 기록을 업데이트합니다.

        """
        if not self.__streaming:
            raise ValueError(
                "Streaming is not enabled. Use 'invoke' method instead or initialize ROSA with streaming=True."
            )

        try:
            final_output = ""
            # 에이전트의 reponse로 부터 events들을 스트림하여 각 결과를 처리
            # Stream events from the agent's response
            async for event in self.__executor.astream_events(
                input={"input": query, "chat_history": self.__chat_history},
                config={"run_name": "Agent"},
                version="v2",
            ):
                # Extract the event type
                kind = event["event"]

                # Handle chat model stream events
                if kind == "on_chat_model_stream":
                    # Extract the content from the event and yield it
                    content = event["data"]["chunk"].content
                    if content:
                        final_output += f" {content}"
                        yield {"type": "token", "content": content}

                # Handle tool start events
                elif kind == "on_tool_start":
                    yield {
                        "type": "tool_start",
                        "name": event["name"],
                        "input": event["data"].get("input"),
                    }

                # Handle tool end events
                elif kind == "on_tool_end":
                    yield {
                        "type": "tool_end",
                        "name": event["name"],
                        "output": event["data"].get("output"),
                    }

                # Handle chain end events
                elif kind == "on_chain_end":
                    if event["name"] == "Agent":
                        chain_output = event["data"].get("output", {}).get("output")
                        if chain_output:
                            final_output = (
                                chain_output  # Override with final output if available
                            )
                            yield {"type": "final", "content": chain_output}

            if final_output:
                self._record_chat_history(query, final_output)
        except Exception as e:
            yield {"type": "error", "content": f"An error occurred: {e}"}

    def _get_executor(self, verbose: bool) -> AgentExecutor:
        """Create and return an executor for processing user inputs and generating responses."""
        executor = AgentExecutor(
            agent=self.__agent,
            tools=self.__tools.get_tools(),
            stream_runnable=self.__streaming,
            verbose=verbose,
            # max_iterations=5,               # 실행 루프를 종료하기 전 최대 단계 수 -> 다섯 단계만 
            # max_execution_time=20,          # 실행 루프에 소요될 수 있는 최대 시간
        )
        return executor

    def _get_agent(self):
        """Create and return an agent for processing user inputs and generating responses."""
        agent = (
            {
                "input": lambda x: x["input"],
                "agent_scratchpad": lambda x: format_to_openai_tool_messages(
                    x["intermediate_steps"]
                ),
                "chat_history": lambda x: x["chat_history"],
            }
            | self.__prompts
            | self.__llm_with_tools
            | OpenAIToolsAgentOutputParser()
        )
        return agent

    def _get_tools(
        self,
        packages: Optional[list],
        tools: Optional[list],
        blacklist: Optional[list],
    ) -> ROSATools:
        """Create a ROSA tools object with the specified ROS version, tools, packages, and blacklist."""
        rosa_tools = ROSATools(blacklist=blacklist)
        if tools:
            rosa_tools.add_tools(tools)
        if packages:
            rosa_tools.add_packages(packages, blacklist=blacklist)
        return rosa_tools

    def _get_prompts(
        self, robot_prompts: Optional[RobotSystemPrompts] = None
    ) -> ChatPromptTemplate:
        """Create a chat prompt template from the system prompts and robot-specific prompts."""
        # Start with default system prompts
        prompts = system_prompts

        # Add robot-specific prompts if provided
        if robot_prompts:
            prompts.append(robot_prompts.as_message())

        template = ChatPromptTemplate.from_messages(
            prompts
            + [
                MessagesPlaceholder(variable_name=self.__memory_key),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name=self.__scratchpad),
            ]
        )
        return template

    def _print_usage(self, cb):
        """Print the token usage if show_token_usage is enabled."""
        if cb and self.__show_token_usage:
            print(f"[bold]Prompt Tokens:[/bold] {cb.prompt_tokens}")
            print(f"[bold]Completion Tokens:[/bold] {cb.completion_tokens}")
            print(f"[bold]Total Cost (USD):[/bold] ${cb.total_cost}")

    def _record_chat_history(self, query: str, response: str):
        """Record the chat history if accumulation is enabled."""
        if self.__accumulate_chat_history:
            self.__chat_history.extend(
                [HumanMessage(content=query), AIMessage(content=response)]
            )
