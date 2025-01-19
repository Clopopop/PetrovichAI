"""
Encapsulates LangGraph logic: creates a StateGraph, registers nodes,
and provides a method to run (invoke) the flow with incoming messages.
"""

import datetime
import random

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, RemoveMessage
from langchain_community.tools import TavilySearchResults
from langgraph.prebuilt import ToolNode

from logger_setup import logger
from config import ConfigManager


class WorkflowController:
    """
    Creates and manages the LangGraph-based workflow.
    """

    def __init__(self, config: ConfigManager):
        self.config = config
        self.memory = MemorySaver()

        # Prepare the system prompt with current date and time
        date_and_time = datetime.datetime.now().strftime("%d.%m.%Y %H:%M")
        self.SYSTEM_PROMPT = (
            "Вы — ПетровичAI, приятный и ненавязчивый собеседник. Вы общаетесь на русском языке, "
            "если только вас прямо не попросят отвечать на другом языке. "
            "Вы отвечаете лаконично, одним-двумя предложениями. "
            f"Сейчас {date_and_time}."
        )

        # Initialize tools and LLM
        search_tool = TavilySearchResults(
            max_results=10,
            include_answer=True,
            include_raw_content=False
        )
        tools = [search_tool]

        self.llm = ChatOpenAI(
            model=self.config.MAIN_WORKFLOW_MODEL,
            api_key=self.config.OPENAI_API_KEY
        ).bind_tools(tools)

        # Build the graph
        self.graph = self._build_graph(tools)
        logger.info("WorkflowController: Initialization complete.")

    def _build_graph(self, tools) -> StateGraph:
        """
        Creates and compiles the StateGraph.
        """
        workflow = StateGraph(MessagesState)
        tool_node = ToolNode(tools)

        # Register nodes
        workflow.add_node("node_llm_query", self._node_llm_query)
        workflow.add_node("tools", tool_node)
        workflow.add_node("node_truncate_message_history", self._node_truncate_message_history)

        # Define edges
        workflow.add_conditional_edges(START, self._bot_should_respond_router, ["node_llm_query", END])
        workflow.add_conditional_edges("node_llm_query", self._tool_router, ["tools", "node_truncate_message_history"])
        workflow.add_edge("tools", "node_llm_query")

        workflow.set_finish_point("node_truncate_message_history")

        return workflow.compile(checkpointer=self.memory)

    def _bot_should_respond_router(self, state: MessagesState) -> str:
        """
        Checks if the bot should respond to the last message.
        """
        from petrovichai import BOT_USERNAME
        last_message = state["messages"][-1]

        should_respond = self._bot_should_respond(last_message.content, BOT_USERNAME)

        logger.info(f"_bot_should_respond_router: last_message='{last_message.content}', respond={should_respond}")
        return "node_llm_query" if should_respond else END

    def _tool_router(self, state: MessagesState) -> str:
        """
        If the LLM wants to call a tool, go to the ToolNode; otherwise, truncate the message history.
        """
        last_message = state["messages"][-1]
        if last_message.tool_calls:
            logger.info("_tool_router: Tool calls detected.")
            return "tools"
        
        logger.info("_tool_router: No tool calls. Truncating message history.")
        return "node_truncate_message_history"

    def _node_llm_query(self, state: MessagesState) -> dict:
        """
        Sends conversation messages to the LLM and returns the AI response.
        """
        messages = state["messages"]

        # Append system prompt if not already present
        if not any(isinstance(msg, SystemMessage) and msg.content == self.SYSTEM_PROMPT for msg in messages):
            messages.append(SystemMessage(self.SYSTEM_PROMPT))

        response = self.llm.invoke(messages)
        logger.info(f"_node_llm_query: LLM response: '{response.content}'")
        return {"messages": [response]}

    def _node_truncate_message_history(self, state: MessagesState) -> dict:
        """
        Keeps only the last MESSAGE_HISTORY_LIMIT messages.
        """
        msgs_to_keep = self.config.MESSAGE_HISTORY_LIMIT
        current_len = len(state["messages"])
        delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-msgs_to_keep]]

        logger.info(
            f"_node_truncate_message_history: Current total messages={current_len}, "
            f"truncating to {msgs_to_keep}."
        )
        return {"messages": delete_messages}

    def _is_bot_mentioned(self, message: str, bot_username: str) -> bool:
        """
        Checks if the message mentions the bot.
        """
        if not message:
            return False
        lower_text = message.lower()
        return any(
            keyword in lower_text
            for keyword in ["петрович", "бот", "bot", f"@{bot_username}"]
        )
    def _bot_should_respond(self, message: str, bot_username: str) -> bool:
        """
        Checks if the bot should respond to the message.
        """
        return random.random() < self.config.RANDOM_RESPONSE_PROBABILITY or self._is_bot_mentioned(message, bot_username)
    
    def bot_should_respond(self, message: str) -> bool:
        """
        Public method to check if the bot should respond to the message.
        """
        from petrovichai import BOT_USERNAME
        return self._bot_should_respond(message, BOT_USERNAME)
    
    def invoke_flow(self, messages_dict: dict, thread_config: dict) -> dict:
        """
        Public method to pass messages into the graph. 
        Returns the result (a dict containing new messages).
        """
        return self.graph.invoke(messages_dict, thread_config)
