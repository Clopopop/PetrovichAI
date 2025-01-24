"""
Encapsulates LangGraph logic: creates a StateGraph, registers nodes,
and provides a method to run (invoke) the flow with incoming messages.
"""

import datetime
import random
import sqlite3

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, RemoveMessage, ToolMessage, AIMessage, HumanMessage
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
        conn = sqlite3.connect("main_workflow_memory.sqlite", check_same_thread=False)
        self.memory = SqliteSaver(conn)

        # Prepare the system prompt with current date and time
        date_and_time = datetime.datetime.now().strftime("%d.%m.%Y %H:%M")
        self.SYSTEM_PROMPT = (
            "Вы — ПетровичAI, приятный и ненавязчивый собеседник. Вы общаетесь на русском языке, "
            "если только вас прямо не попросят отвечать на другом языке. "
            "Вы отвечаете лаконично, одним-двумя предложениями. "
            "Используйте инструмент TavilySearchResults для поиска информации в интернете. "
            f"Сейчас {date_and_time}."
        )

        self.SYSTEM_PROMPT_SHOULD_RESPOND = self.SYSTEM_PROMPT + (
            "Ответом на это сообщение является оценка вероятности того, что ПетровичAI вовлечен в диалог "
            "и от него ожидается ответ, несмотря на то что его имя может быть не упомянуто в сообщении. "
            "Дай ответ в формате одного вещественного числа, с точностью 2 знака после запятой (например 0.52). "
            "Ответ должен быть в диапазоне от 0.0 до 0.99. "
            "Если ты считаешь, что ПетровичAI не должен отвечать на это сообщение, введи 0.0. "
            "Если ты считаешь, что ПетровичAI должен отвечать на это сообщение, введи 0.99. "
            "Отвечай только в этом формате и не выводи никаких других символов. "
            "Возможные причины, по которым ПетровичAI может решить отвечать на сообщение, включают в себя, "
            "но не ограничиваются: "
            "Сообщение содержит упоминание бота или его имени; "
            "Сообщение задаёт вопрос, на который бот знает ответ или может нагуглить; "
            "Сообщение связано с контекстом предыдущего диалога, где бот участвовал; "
            "Явная необходимость ответа по смыслу или обращению. "
            "Далее приведена история сообщений."
        )

        # Initialize tools and LLM
        search_tool = TavilySearchResults(
            max_results=3,
            include_answer=True,
            include_raw_content=False
        )
        tools = [search_tool]

        self.llmMain = ChatOpenAI(
            model=self.config.MAIN_WORKFLOW_MODEL,
            api_key=self.config.OPENAI_API_KEY
        ).bind_tools(tools)

        self.llmShouldReply = ChatOpenAI(
            model=self.config.SHOULD_RESPOND_MODEL,
            api_key=self.config.OPENAI_API_KEY
        )

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
        workflow.add_node("node_truncate_message_history_phase1", self._node_truncate_message_history_phase1)
        workflow.add_node("node_truncate_message_history_phase2", self._node_truncate_message_history_phase2)

        # Define edges
        workflow.add_conditional_edges(START, self._bot_should_respond_router, ["node_llm_query", "node_truncate_message_history_phase1"])
        workflow.add_conditional_edges("node_llm_query", self._tool_router, ["tools", "node_truncate_message_history_phase1"])
        workflow.add_edge("tools", "node_llm_query")
        workflow.add_edge("node_truncate_message_history_phase1", "node_truncate_message_history_phase2")

        return workflow.compile(checkpointer=self.memory)

    def _bot_should_respond_router(self, state: MessagesState) -> str:
        """
        Checks if the bot should respond to the last message.
        """
        from petrovichai import BOT_USERNAME
        last_message = state["messages"][-1]

        should_respond = self._bot_should_respond(state, BOT_USERNAME)

        logger.info(f"_bot_should_respond_router: last_message='{last_message.content}', respond={should_respond}")
        return "node_llm_query" if should_respond else "node_truncate_message_history_phase1"

    def _tool_router(self, state: MessagesState) -> str:
        """
        If the LLM wants to call a tool, go to the ToolNode; otherwise, truncate the message history.
        """
        last_message = state["messages"][-1]
        if last_message.tool_calls:
            logger.info("_tool_router: Tool calls detected.")
            return "tools"
        
        logger.info("_tool_router: No tool calls. Truncating message history.")
        return "node_truncate_message_history_phase1"

    def _node_llm_query(self, state: MessagesState) -> dict:
        """
        Sends conversation messages to the LLM and returns the AI response.
        """
        messages = state["messages"]

        # Append system prompt if not already present
        if not any(isinstance(msg, SystemMessage) and msg.content == self.SYSTEM_PROMPT for msg in messages):
            messages.append(SystemMessage(self.SYSTEM_PROMPT))

        response = self.llmMain.invoke(messages)
        logger.info(f"_node_llm_query: LLM response: '{response.content}'")
        return {"messages": [response]}

    def _node_truncate_message_history_phase1(self, state: MessagesState) -> dict:
        """
        Removes all tool and system messages from the state.
        """

        # delete all tool and system messages from the state
        deleted_tool_messages = [RemoveMessage(id=m.id) for m in state["messages"] if isinstance(m, ToolMessage)]
        deleted_system_messages = [RemoveMessage(id=m.id) for m in state["messages"] if isinstance(m, SystemMessage)]
        
        #delete assistant messages with tool calls
        deleted_toolcalls_messages = [RemoveMessage(id=m.id) for m in state["messages"] if isinstance(m, AIMessage) and m.tool_calls]

        deleted_messages = deleted_tool_messages + deleted_system_messages + deleted_toolcalls_messages

        return {"messages": deleted_messages}

    def _node_truncate_message_history_phase2(self, state: MessagesState) -> dict:
        """
        Keeps only the last MESSAGE_HISTORY_LIMIT messages.
        """
        msgs_to_keep = self.config.MESSAGE_HISTORY_LIMIT
        deleted_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-msgs_to_keep]]

        return {"messages": deleted_messages}

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
    
    def _bot_should_respond(self, state: MessagesState, bot_username: str) -> bool:
        """
        Checks if the bot should respond to the message.
        """

        HISTORY_LENGTH_TO_ANALYZE = 6

        last_message = state["messages"][-1].content

        # First make simple checks like random probability and bot mention
        if random.random() < self.config.RANDOM_RESPONSE_PROBABILITY or self._is_bot_mentioned(last_message, bot_username):
            return True
        
        # Check if the LLM thinks the bot should respond
        message_history = state["messages"][-HISTORY_LENGTH_TO_ANALYZE:]
        
        # Remove system messages from the history
        messages = [msg for msg in message_history if not isinstance(msg, SystemMessage)]

        # Add specific system prompt to analyse if LLM should respond on the last message
        # adding it twice make the results better \_(o.o)_/
        messages = [SystemMessage(self.SYSTEM_PROMPT_SHOULD_RESPOND)] + messages + [SystemMessage(self.SYSTEM_PROMPT_SHOULD_RESPOND)]

        #invoke LLM. Answer should be a float.
        response = self.llmShouldReply.invoke(messages)

        #translate response to float
        try:
            reply_probability = float(response.content)
        except ValueError:
            logger.error(f"Invalid response from LLM: {response.content}")
            return False

        logger.info(f"_bot_should_respond: LLM response: {reply_probability}, while threshold is {self.config.LLM_DECISSION_TO_RESPOND_THRESHOLD}")
        return reply_probability > self.config.LLM_DECISSION_TO_RESPOND_THRESHOLD

    
    def bot_should_respond(self, message: str) -> bool:
        """
        Public method to check if the bot should respond to the message.
        """
        from petrovichai import BOT_USERNAME

        if not message:
            logger.error("bot_should_respond: Empty message.")
            return False

        # convert the string message into a state for further processing
        messages_state = MessagesState(messages=[HumanMessage(content=message)])
        return self._bot_should_respond(messages_state, BOT_USERNAME)        
    
    def invoke_flow(self, messages_dict: dict, thread_config: dict) -> dict:
        """
        Public method to pass messages into the graph. 
        Returns the result (a dict containing new messages).
        """
        return self.graph.invoke(messages_dict, thread_config)
