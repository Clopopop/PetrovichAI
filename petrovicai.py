import asyncio
import os
import random
from configparser import ConfigParser
from collections import deque
from aiogram import Bot, Dispatcher, types
from aiogram.types import Message
import openai
import base64

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
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages
                )
        else:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": PETROVICH_CONTEXT}] + prompt
            )
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Helper function to check if the bot is mentioned
async def is_bot_mentioned(message: Message):
    if message.text is None:
        return False
    bot_username = (await bot.get_me()).username.lower()
    return "петрович" in message.text.lower() or f"@{bot_username}" in message.text.lower()

# Handler for incoming messages
@dp.message()
async def handle_message(message: Message):
    chat_id = message.chat.id

    # Initialize message history for the chat if it doesn't exist
    if chat_id not in chat_histories:
        chat_histories[chat_id] = deque(maxlen=MESSAGE_HISTORY_LIMIT)

    # Save the message to the history
    image_path = None
    if message.content_type == 'text':
        chat_histories[chat_id].append({"role": "user", "content": message.text})
    elif message.content_type == 'photo':
        file_id = message.photo[-1].file_id
        file_info = await bot.get_file(file_id)
        image_path = f"temp_{file_id}.jpg"
        await bot.download_file(file_info.file_path, image_path)
        chat_histories[chat_id].append({"role": "user", "content": "The user sent an image."})
    elif message.content_type == 'document':  # Consider documents
        chat_histories[chat_id].append({"role": "user", "content": "The user sent a file."})

    # Check if the bot is directly mentioned or called by a similar name
    is_direct_mention = await is_bot_mentioned(message)

    # Decide whether the bot should respond to a random message
    should_respond = random.random() < RANDOM_RESPONSE_PROBABILITY

    # Condition for response: direct mention or random choice
    if is_direct_mention or should_respond:
        # Form the prompt from the message history
        prompt = list(chat_histories[chat_id])
        if message.content_type == 'text':
            prompt.append({"role": "user", "content": message.text})
        elif message.content_type == 'photo':
            prompt.append({"role": "user", "content": "Комментарий к изображению: Пользователь отправил изображение."})

        response = await get_openai_response(prompt, image_path)
        await message.reply(response)

        # Clean up the temporary image file if it was downloaded
        if image_path:
            os.remove(image_path)

async def main():
    print("Bot is running!")
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot, skip_updates=True)

if __name__ == "__main__":
    asyncio.run(main())
