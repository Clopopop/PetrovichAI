"""
Main entry point of the application. Creates all classes and runs the bot.
"""

import asyncio
from aiogram import Bot, Dispatcher
from aiogram.types import Message

from logger_setup import logger
from config import CONFIG
from workflow_controller import WorkflowController
from telegram_message_handler import TelegramMessageHandler

BOT_USERNAME = None  # Global variable for storing the bot's username


class BotApplication:
    """
    Orchestrates the entire application:
    - Initializes Bot and Dispatcher
    - Starts Aiogram polling
    - Registers handlers
    """

    def __init__(self):
        logger.info("BotApplication: Initializing...")

        # Initialize Telegram Bot and Dispatcher
        self.bot = Bot(token=CONFIG.TELEGRAM_TOKEN)
        self.dispatcher = Dispatcher()

        # Create the LangGraph workflow controller
        self.workflow_controller = WorkflowController(CONFIG)

        # Create the main message handler
        self.message_handler = TelegramMessageHandler(self.workflow_controller)

        logger.info("BotApplication: Initialization complete.")

    async def initialize_bot_username(self):
        """
        Gets the bot's username from Telegram for mention-related logic.
        """
        global BOT_USERNAME
        me = await self.bot.get_me()
        BOT_USERNAME = me.username.lower()
        logger.info(f"BotApplication: BOT_USERNAME set to: {BOT_USERNAME}")

    def register_handlers(self):
        """
        Registers Aiogram handlers for all incoming messages.
        """
        @self.dispatcher.message()
        async def handle_message(msg: Message):
            await self.message_handler.route_incoming_message(msg, self.bot)

    async def start(self):
        """
        Main asynchronous method to start the bot.
        """
        await self.initialize_bot_username()
        print("Bot is running!")

        # Register handlers
        self.register_handlers()

        # Clear any existing queued updates and begin polling
        await self.bot.delete_webhook(drop_pending_updates=True)
        await self.dispatcher.start_polling(self.bot, skip_updates=True)


async def main():
    """
    Launches the entire application.
    """
    app = BotApplication()
    await app.start()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user.")
        print("Bot stopped by user")
