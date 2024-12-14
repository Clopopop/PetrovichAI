import asyncio
import os
import random
from configparser import ConfigParser
from collections import deque
from aiogram import Bot, Dispatcher, types
from aiogram.types import Message
import openai
import base64
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='bot_log.log', filemode='a')
logger = logging.getLogger()

# Read tokens from config file
config = ConfigParser()
config.read("config.ini")

TELEGRAM_TOKEN = config.get("tokens", "TELEGRAM_TOKEN")
OPENAI_API_KEY = config.get("tokens", "OPENAI_API_KEY")

# Initialize tokens
bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Settings
RANDOM_RESPONSE_PROBABILITY = 1  # Probability of responding to a random message (0.0 - never, 1.0 - always)
MESSAGE_HISTORY_LIMIT = 30  # Number of recent messages to consider

# Message history for each chat
chat_histories = {}

# Context for ПетровичAI
PETROVICH_CONTEXT = (
    "Вы — ПетровичAI, уникальная скуфмодель, обладающая выдающимся умом и сообразительностью. "
    "Вы общаетесь на русском языке, если только вас прямо не попросят отвечать на другом языке. "
    "Ваш стиль — остроумный, с логичным и рациональным подходом к каждой теме. "
    "Когда уместно, вы добавляете нотку юмора, чтобы сделать общение более живым и запоминающимся. "
    "Вы можете поворчать, проявить недовольство и начать спорить. "
    "Вы не говорите что созданы для помощи, не предлагаете помощь. "
    "Ваши ответы всегда умны, но никогда не выходят за рамки контекста, оставляя ощущение профессионализма и дружелюбия. "
    "Ваш девиз: «Логика — основа, остроумие — украшение».")

# Function to get a response from OpenAI API
async def get_openai_response(prompt, image_path=None):
    try:
        if image_path:
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode()
                messages = [
                    {"role": "system", "content": PETROVICH_CONTEXT},
                    {"role": "user", "content": [ 
                      {"type": "text", "text": "Прокомментируй изображение."},
                      {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                    ]}
                ]
                logger.info(f"Sending to OpenAI (image): {messages}")
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages
                )
        else:
            logger.info(f"Sending to OpenAI (text): {prompt}")
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": PETROVICH_CONTEXT}] + prompt
            )
        logger.info(f"OpenAI response: {response.choices[0].message.content}")
        return response.choices[0].message.content
    except Exception as e:
        error_message = f"Error communicating with OpenAI: {str(e)}"
        logger.error(error_message)
        # Do not pass the error to Telegram, return a generic error message
        return None

# Helper function to check if the bot is mentioned
async def is_bot_mentioned(message: Message):
    if message.text is None:
        return False
    bot_username = (await bot.get_me()).username.lower()
    return "петрович" in message.text.lower() or f"@{bot_username}" in message.text.lower()

# Update message history by maintaining order and limiting size
def update_message_history(chat_id, role, content):
    if chat_id not in chat_histories:
        chat_histories[chat_id] = deque(maxlen=MESSAGE_HISTORY_LIMIT)
    chat_histories[chat_id].append({"role": role, "content": content})

# Handler for incoming messages
@dp.message()
async def handle_message(message: Message):
    chat_id = message.chat.id
    user_name = message.from_user.full_name or "Пользователь"

    # Save the message to the history
    image_path = None
    if message.content_type == 'text':
        update_message_history(chat_id, "user", f"{user_name}: {message.text}")
    elif message.content_type == 'photo':
        file_id = message.photo[-1].file_id
        file_info = await bot.get_file(file_id)
        image_path = f"temp_{file_id}.jpg"
        await bot.download_file(file_info.file_path, image_path)
        update_message_history(chat_id, "user", f"{user_name} отправил изображение.")
    elif message.content_type == 'document':  # Consider documents
        update_message_history(chat_id, "user", f"{user_name} отправил файл.")

    # Check if the bot is directly mentioned or called by a similar name
    is_direct_mention = await is_bot_mentioned(message)

    # Decide whether the bot should respond to a random message
    should_respond = random.random() < RANDOM_RESPONSE_PROBABILITY

    # Condition for response: direct mention or random choice
    if is_direct_mention or should_respond:
        # Form the prompt from the message history
        prompt = list(chat_histories[chat_id])
        response = await get_openai_response(prompt, image_path)
        if response:  # Only reply if a valid response is received
            await message.reply(response)
            update_message_history(chat_id, "assistant", response)
        else:
            await message.reply("Извините, я не могу обработать ваш запрос сейчас.")

        # Clean up the temporary image file if it was downloaded
        if image_path:
            try:
                os.remove(image_path)
            except Exception as e:
                logger.error(f"Error deleting temporary file: {str(e)}")

async def main():
    print("Bot is running!")
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot, skip_updates=True)

if __name__ == "__main__":
    asyncio.run(main())
