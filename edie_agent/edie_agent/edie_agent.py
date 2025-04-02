import asyncio
import os
from datetime import datetime

import dotenv
import pyinputplus as pyip
import rclpy
from langchain.agents import tool, Tool
from rich.console import Console
from rich.console import Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text


from rosa import ROSA
from .toolbox import move as move_tool
from .utils.help import get_help
from .utils.llm import get_llm
from .prompts.prompt import get_prompts


@tool
def cool_edie_tool():
    """A cool EDIE tool that doesn't really do anything."""
    return "This is a cool edie tool! It doesn't do anything, but it's cool."


class EDIEAgent(ROSA):
    def __init__(self, streaming: bool = False, verbose: bool = True):
        self.__blacklist = ["docker"]
        self.__prompts = get_prompts()
        self.__llm = get_llm(streaming=streaming)
        self.__streaming = streaming



        super().__init__(
            llm=self.__llm,
            tools=[cool_edie_tool],
            tool_packages=[move_tool],
            blacklist=self.__blacklist,
            prompts=self.__prompts,
            verbose=verbose,
            accumulate_chat_history=True,
            streaming=streaming,
        )

        self.examples = [
            "Give me a ROS tutorial.",
            "Show me how to move the edie forward.",
            "Give me a list of nodes, topics, services, params, and log files.",
        ]

        self.command_handler = {
            "help": lambda: self.submit(get_help(self.examples)),
            "examples": lambda: self.submit(self.choose_example()),
            "clear": lambda: self.clear(),
        }


    @property
    def greeting(self):
        greeting = Text(
            "\n안녕하세요! 저는 EDIE입니다 🤖. 만나서 반가워요.\n"
        )
        greeting.stylize("frame bold blue")
        greeting.append(
            f"Try {', '.join(self.command_handler.keys())} or exit.",
            style="italic",
        )
        return greeting

    def choose_example(self):
        """예제 목록에서 사용자의 선택을 가져옵니다."""
        return pyip.inputMenu(
            self.examples,
            prompt="\nEnter your choice and press enter: \n",
            numbered=True,
            blank=False,
            timeout=60,
            default="1",
        )

    async def clear(self):
        """채팅 기록을 초기화합니다."""
        self.clear_chat()
        self.last_events = []
        self.command_handler.pop("info", None)
        os.system("clear")

    def get_input(self, prompt: str):
        """콘솔에서 사용자 입력을 받습니다."""
        return pyip.inputStr(prompt, default="help")

    async def run(self):
        """
        EDIEAgent의 메인 상호작용 루프를 실행합니다.
        콘솔 인터페이스를 초기화하고 사용자 입력을 지속적으로 처리합니다.
        'help', 'examples', 'clear', 'exit' 명령 및 사용자 질의를 처리합니다.
        """
        await self.clear()
        console = Console()

        while True:
            console.print(self.greeting)
            user_input = self.get_input("> ")

            # 특수 명령 처리
            if user_input == "exit":
                break
            elif user_input in self.command_handler:
                await self.command_handler[user_input]()
            else:
                await self.submit(user_input)

    async def submit(self, query: str):
        if self.__streaming:
            await self.stream_response(query)
        else:
            self.print_response(query)

    def print_response(self, query: str):
        """
        에이전트에 질의를 제출하고, 응답을 콘솔에 출력합니다.
        """
        response = self.invoke(query)
        console = Console()
        content_panel = None

        with Live(
            console=console, auto_refresh=True, vertical_overflow="visible"
        ) as live:
            content_panel = Panel(
                Markdown(response), title="Final Response", border_style="green"
            )
            live.update(content_panel, refresh=True)

    async def stream_response(self, query: str):
        """
        에이전트의 응답을 실시간 스트리밍 방식으로 출력합니다.
        """
        console = Console()
        content = ""
        self.last_events = []

        panel = Panel("", title="Streaming Response", border_style="green")

        with Live(panel, console=console, auto_refresh=False) as live:
            async for event in self.astream(query):
                event["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                if event["type"] == "token":
                    content += event["content"]
                    panel.renderable = Markdown(content)
                    live.refresh()
                elif event["type"] in ["tool_start", "tool_end", "error"]:
                    self.last_events.append(event)
                elif event["type"] == "final":
                    content = event["content"]
                    if self.last_events:
                        panel.renderable = Markdown(
                            content
                            + "\n\nType 'info' for details on how I got my answer."
                        )
                    else:
                        panel.renderable = Markdown(content)
                    panel.title = "Final Response"
                    live.refresh()

        if self.last_events:
            self.command_handler["info"] = self.show_event_details
        else:
            self.command_handler.pop("info", None)

    async def show_event_details(self):
        """
        마지막 질의에서 발생한 이벤트에 대한 상세 정보를 표시합니다.
        """
        console = Console()

        if not self.last_events:
            console.print("[yellow]No events to display.[/yellow]")
            return
        else:
            console.print(Markdown("# Tool Usage and Events"))

        for event in self.last_events:
            timestamp = event["timestamp"]
            if event["type"] == "tool_start":
                console.print(
                    Panel(
                        Group(
                            Text(f"Input: {event.get('input', 'None')}"),
                            Text(f"Timestamp: {timestamp}", style="dim"),
                        ),
                        title=f"Tool Started: {event['name']}",
                        border_style="blue",
                    )
                )
            elif event["type"] == "tool_end":
                console.print(
                    Panel(
                        Group(
                            Text(f"Output: {event.get('output', 'N/A')}"),
                            Text(f"Timestamp: {timestamp}", style="dim"),
                        ),
                        title=f"Tool Completed: {event['name']}",
                        border_style="green",
                    )
                )
            elif event["type"] == "error":
                console.print(
                    Panel(
                        Group(
                            Text(f"Error: {event['content']}", style="bold red"),
                            Text(f"Timestamp: {timestamp}", style="dim"),
                        ),
                        border_style="red",
                    )
                )
            console.print()

        console.print("[bold]End of events[/bold]\n")


def main():
    # .env 파일 로드
    dotenv.load_dotenv(dotenv.find_dotenv())

    # ROS2에서는 파라미터 서버 대신 환경변수나 명령줄 인자를 사용합니다.
    streaming = os.getenv("STREAMING", "False").lower() in ["true", "1", "yes"]
    edie_agent = EDIEAgent(verbose=True, streaming=streaming)

    asyncio.run(edie_agent.run())


if __name__ == "__main__":
    rclpy.init(args=None)
    try:
        main()
    finally:
        rclpy.shutdown()


# ros2 run edie_agent edie_agent