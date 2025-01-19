"""
Class that handles incoming Telegram messages. Uses WorkflowController for decisions.
"""

import os
import base64
from aiogram.types import Message
from langchain_core.messages import HumanMessage

from logger_setup import logger
from workflow_controller import WorkflowController


class TelegramMessageHandler:
    """
    Receives and processes incoming Telegram messages,
    determining how the bot should respond depending on content type.
    """

    def __init__(self, workflow_controller: WorkflowController):
        self.workflow_controller = workflow_controller

    async def route_incoming_message(self, telegram_message: Message, bot):
        """
        Main method called by Aiogram when a new message arrives.
        
        :param telegram_message: The Telegram message object
        :param bot: The Aiogram Bot instance for file downloads, etc.
        """
        chat_id = str(telegram_message.chat.id)
        user_name = telegram_message.from_user.full_name or "User"
        logger.info(
            f"route_incoming_message: Received message in chat {chat_id} from {user_name}, type={telegram_message.content_type}"
        )

        conversation_thread_config = {"configurable": {"thread_id": chat_id}}
        response = None

        # --- Handle text messages ---
        if telegram_message.content_type == "text":
            input_text = f"{user_name}: {telegram_message.text}"
            output = self._handle_text_message(input_text, conversation_thread_config)
            if output and output["messages"][-1].type == "ai":
                response = output["messages"][-1].content
            else:
                return

        # --- Handle voice messages ---
        elif telegram_message.content_type == "voice":
            response = await self._handle_voice_message(telegram_message, user_name)

        # --- Handle photos or documents ---
        elif telegram_message.content_type in ["photo", "document"]:
            # decide whether to reply on the message with image according to the common logic
            if not self.workflow_controller.bot_should_respond(telegram_message.caption):
                logger.info("_handle_image_or_document_message: Bot not mentioned in caption.")
                return 

            response = await self._handle_image_or_document_message(telegram_message, user_name, bot)

        # --- Send the response or default apology ---
        if response:
            logger.info(f"route_incoming_message: Sending response to user: {response[:70]}...")
            await telegram_message.reply(response)
        else:
            logger.info("route_incoming_message: No response formed. Sending apology message.")
            await telegram_message.reply("Извините, я не могу обработать ваш запрос сейчас.")

    def _handle_text_message(self, input_text: str, conversation_thread_config: dict) -> dict:
        """
        Passes the text message through WorkflowController.
        Returns the graph output (a dict) if any.
        """
        input_message = HumanMessage(input_text)
        output = self.workflow_controller.invoke_flow(
            {"messages": input_message}, 
            conversation_thread_config
        )
        return output

    async def _handle_voice_message(self, telegram_message: Message, user_name: str) -> str:
        """
        Processes voice messages (currently not implemented).
        """
        logger.info(f"_handle_voice_message: Voice message received from {user_name}, not supported yet.")
        return "Голосовые сообщения пока не поддерживаются."

    async def _handle_image_or_document_message(self, telegram_message: Message, user_name: str, bot) -> str:
        """
        Downloads an image or document, encodes it in Base64, and sends it to the LLM.
        :return: The LLM response or an empty string if no file ID.
        """

        file_id = None
        if telegram_message.content_type == "photo":
            file_id = telegram_message.photo[-1].file_id
        elif telegram_message.content_type == "document":
            file_id = telegram_message.document.file_id

        if not file_id:
            logger.info("_handle_image_or_document_message: Unable to get file_id.")
            return ""

        file_info = await bot.get_file(file_id)
        image_path = f"temp_{file_id}.jpg"

        await bot.download_file(file_info.file_path, image_path)
        logger.info(f"_handle_image_or_document_message: File downloaded to {image_path}")

        response_text = None
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode()
            output = self.workflow_controller.llm.invoke(
                [
                    (
                        "human",
                        [
                            {"type": "text", "text": f"{user_name}: {telegram_message.caption or ''}"},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                        ],
                    ),
                    ("system", self.workflow_controller.SYSTEM_PROMPT),
                ]
            )
            response_text = output.content

        # Remove the temporary file
        try:
            os.remove(image_path)
            logger.info(f"_handle_image_or_document_message: Removed temporary file {image_path}")
        except Exception as e:
            logger.error(f"Error removing temporary file {image_path}: {str(e)}")

        return response_text
