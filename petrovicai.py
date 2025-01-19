import asyncio
import os
import random
from configparser import ConfigParser
from aiogram import Bot, Dispatcher, types
from aiogram.types import Message
import base64
import logging
import datetime

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, RemoveMessage
from langchain_community.tools import TavilySearchResults
from langgraph.prebuilt import ToolNode

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='bot_log.log', filemode='a')
logger = logging.getLogger()

# Read tokens from config file
config = ConfigParser()
config.read("config.ini")
TELEGRAM_TOKEN = config.get("tokens", "TELEGRAM_TOKEN")
OPENAI_API_KEY = config.get("tokens", "OPENAI_API_KEY")
TAVILY_API_KEY = config.get("tokens", "TAVILY_API_KEY")
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

RANDOM_RESPONSE_PROBABILITY = config.getfloat("settings", "RANDOM_RESPONSE_PROBABILITY")
MESSAGE_HISTORY_LIMIT = config.getint("settings", "MESSAGE_HISTORY_LIMIT")

# Initialize bot and dispatcher
bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()

# Initialize MemorySaver
memory = MemorySaver()

date_and_time = datetime.datetime.now().strftime("%d.%m.%Y %H:%M")
SYSTEM_PROMPT = (
    "Вы — ПетровичAI, приятный и ненавязчивый собеседник. Вы общаетесь на русском языке, "
    "если только вас прямо не попросят отвечать на другом языке. "
    "Вы отвечаете лаконично, одним-двумя предложениями. "
    f"Сейчас {date_and_time}."
)

search_tool = TavilySearchResults(max_results=10, include_answer=True, include_raw_content=False)
tools = [search_tool]
llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY).bind_tools(tools)

def bot_should_respond_router(state: MessagesState):
    # Check if the bot should respond to the message
    messages = state["messages"]
    last_message = messages[-1]
    return "node_llm_query" if bot_should_respond(last_message.content) else END

def tool_router(state: MessagesState):
    messages = state["messages"]
    print("Messages: ")
    print(messages)

    last_message = messages[-1]
    return "tools" if last_message.tool_calls else "node_truncate_message_history"

def node_llm_query(state: MessagesState):
    messages = state["messages"]

    # Add system prompt if not present
    if not any(isinstance(msg, SystemMessage) and msg.content == SYSTEM_PROMPT for msg in messages):
        messages.append(SystemMessage(SYSTEM_PROMPT))
        
    response = llm.invoke(messages)
    return {"messages": [response]}

def node_truncate_message_history(state: MessagesState):
    # Delete all but the last MESSAGE_HISTORY_LIMIT messages from the `messages` list in the state 
    delete_messages = [RemoveMessage(id=m.id) for m in state['messages'][:-MESSAGE_HISTORY_LIMIT]]
    return {"messages": delete_messages}

tool_node = ToolNode(tools)
workflow = StateGraph(MessagesState)
workflow.add_node("node_llm_query", node_llm_query)
workflow.add_node("tools", tool_node)
workflow.add_node("node_truncate_message_history", node_truncate_message_history)

workflow.add_conditional_edges(START, bot_should_respond_router, ["node_llm_query", END])
workflow.add_conditional_edges("node_llm_query", tool_router, ["tools", "node_truncate_message_history"])
workflow.add_edge("tools", "node_llm_query")
workflow.set_finish_point("node_truncate_message_history")
graph = workflow.compile(checkpointer=memory)

# Initialize bot username
bot_username = None

async def initialize_bot_username():
    global bot_username
    bot_username = (await bot.get_me()).username.lower()

# Helper function to check if the bot should respond to the message
def bot_should_respond(last_message: str) -> bool:
    return random.random() < RANDOM_RESPONSE_PROBABILITY or is_bot_mentioned(last_message)

# Helper function to check if the bot is mentioned
def is_bot_mentioned(message: str) -> bool:
    if message is None:
        return False
    return any(keyword in message.lower() for keyword in ["петрович", "бот", f"@{bot_username}"])

# Processing of messages with images
async def process_message_with_image(telegramMessage: Message, llm : ChatOpenAI = None):
    user_name = telegramMessage.from_user.full_name or "User"
    file_id = None

    if telegramMessage.content_type == 'photo':
        file_id = telegramMessage.photo[-1].file_id
    if telegramMessage.content_type == 'document':
        # for now process documents as images. For example, PNGs are received as documents
        file_id = telegramMessage.document.file_id

    if file_id:
        file_info = await bot.get_file(file_id)
        image_path = f"temp_{file_id}.jpg"
        await bot.download_file(file_info.file_path, image_path)

        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode()

            output = llm.invoke(
                [
                    (
                        "human",
                        [
                            {"type": "text", "text": f"{user_name}: {telegramMessage.text}."},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                        ],
                    ), 
                    SystemMessage(SYSTEM_PROMPT),
                ]
            )
            response = output.content

        if image_path:
            try:
                os.remove(image_path)
            except Exception as e:
                logger.error(f"Error deleting temporary file: {str(e)}")
    
    return response

# Processing of messages with voice messages
async def process_message_with_voice(telegramMessage: Message, llm : ChatOpenAI = None):
    file_id = None
    # TODO: process voice messages
    return "Голосовые сообщения пока не поддерживаются."

# Handler for incoming Telegram messages
@dp.message()
async def handle_message(telegramMessage: Message):
    chat_id = str(telegramMessage.chat.id)
    user_name = telegramMessage.from_user.full_name or "User"
    image_path = None

    #langgraph thread id is chat_id
    conversation_thread_config = {"configurable": {"thread_id": chat_id}}

    response = None

    if telegramMessage.content_type == 'text':
        input_message_text = f"{user_name}: {telegramMessage.text}"
        input_message = HumanMessage(input_message_text)
        output = graph.invoke({"messages": input_message}, conversation_thread_config)

        # stop if the last message is not from AI, so that LLM was not queried.
        if output["messages"][-1].type != "ai":
            return
        
        response = output["messages"][-1].content

    # process chat messages with attachments manually, without langgraph
    elif telegramMessage.content_type == 'voice':
        # always try to transcript voice messages
        response = await process_message_with_voice(telegramMessage, llm)

    # decide whether to reply on the message with image according to the common logic
    elif telegramMessage.content_type in ['photo', 'document']:
        if bot_should_respond(telegramMessage.caption):
            response = await process_message_with_image(telegramMessage, llm)
        else:
            return
    
    if response:
        await telegramMessage.reply(response)
    else:
        await telegramMessage.reply("Извините, я не могу обработать ваш запрос сейчас.")



async def main():
    await initialize_bot_username()
    print("Bot is running!")
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot, skip_updates=True)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Bot stopped by user")
