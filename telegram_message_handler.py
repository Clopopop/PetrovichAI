"""
Class that handles incoming Telegram messages. Uses WorkflowController for decisions.
"""

import os
import base64
from aiogram.types import Message
from langchain_core.messages import HumanMessage

from logger_setup import logger
from workflow_controller import WorkflowController

from config import CONFIG
from transcriber import Transcriber


# Maximum video file size in bytes
MAX_VIDEO_SIZE = 50 * 1024 * 1024  # 50 MB


class TelegramMessageHandler:
    """
    Receives and processes incoming Telegram messages,
    determining how the bot should respond depending on content type.
    """

    def __init__(self, workflow_controller: WorkflowController):
        self.workflow_controller = workflow_controller
        self.transcriber = Transcriber(CONFIG.OPENAI_API_KEY)


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
            transcription = await self._handle_voice_message(telegram_message, bot)

            # insert the transcription into the workflow and generate a response if applicable
            if transcription:
                input_message_text = f"Голосовое сообщение от {telegram_message.from_user.full_name}: {transcription}"

                # post the transcription as a text message to the chat
                await telegram_message.reply(input_message_text)

                # process the transcription as a text message to store it in the workflow and generate a response
                output = self._handle_text_message(input_message_text, conversation_thread_config)
                if output and output["messages"][-1].type == "ai":
                    response = output["messages"][-1].content
                else:
                    return

        # --- Handle video messages by transcribing the text ---
        elif telegram_message.content_type == "video":
            await self._handle_video_message(telegram_message, bot, conversation_thread_config)

            # no response are generated for video messages, the transcption is only stored in the workflow
            return

        # --- Handle photos or documents ---
        elif telegram_message.content_type in ["photo", "document"]:
            # decide whether to reply on the message with image according to the common logic
            if not self.workflow_controller.bot_should_respond(telegram_message.caption):
                logger.info("_handle_image_or_document_message: Bot should not respond.")
                return 

            response = await self._handle_image_or_document_message(telegram_message, user_name, bot)

        # --- Send the response or default apology ---
        if response:
            logger.info(f"route_incoming_message: Sending response to user: {response[:70]}...")
            await telegram_message.reply(response)
        else:
            logger.info("route_incoming_message: No response formed.")
#            await telegram_message.reply("Извините, я не могу обработать ваш запрос сейчас.")

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

    async def _handle_voice_message(self, telegram_message: Message, bot) -> str:
        """
        Transcribe voice messages.
        """
        if telegram_message.content_type != "voice":
            logger.error("_handle_voice_message: Invalid content type.")
            return None
        
        file_id = telegram_message.voice.file_id

        if not file_id:
            logger.error("_handle_voice_message: Unable to get file_id.")
            return None
        
        file_info = await bot.get_file(file_id)
        voice_path = f"temp_{file_id}.ogg"

        await bot.download_file(file_info.file_path, voice_path)
        logger.info(f"_handle_voice_message: File downloaded to {voice_path}")

        transcription = self.transcriber.transcribe(voice_path)
        logger.info(f"_handle_voice_message: Transcription: {transcription}")

        # Remove the temporary file
        try:
            os.remove(voice_path)
            logger.info(f"_handle_voice_message: Removed temporary file {voice_path}")
        except Exception as e:
            logger.error(f"Error removing temporary file {voice_path}: {str(e)}")

        return transcription

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
            output = self.workflow_controller.llmMain.invoke(
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

    async def _handle_video_message(self, telegram_message: Message, bot, conversation_thread_config: dict) -> str:
        """
        Transcribe video messages.
        """
        if telegram_message.content_type != "video":
            logger.error("_handle_video_message: Invalid content type.")
            return None
        
        file_id = telegram_message.video.file_id

        if not file_id:
            logger.error("_handle_video_message: Unable to get file_id.")
            return None
        
        try:
            file_info = await bot.get_file(file_id)
        except Exception as e:
            logger.error(f"_handle_video_message: Failed to get file info for video {file_id}: {e}")
            return None
        if file_info.file_size > MAX_VIDEO_SIZE:
            logger.error(f"_handle_video_message: Video file is too large: {file_info.file_size} bytes.")
            return None

        video_path = f"temp_{file_id}.mp4"

        await bot.download_file(file_info.file_path, video_path)
        logger.info(f"_handle_video_message: File downloaded to {video_path}")

        try:
            transcription = self.transcriber.transcribe_video(video_path)
        except Exception as e:
            logger.error(f"_handle_video_message: Error during transcription: {e}")
            transcription = None

        logger.info(f"_handle_video_message: Transcription: {transcription}")

        # Remove the temporary file
        try:
            os.remove(video_path)
            logger.info(f"_handle_video_message: Removed temporary file {video_path}")
        except Exception as e:
            logger.error(f"Error removing temporary file {video_path}: {str(e)}")

        # insert the transcription into the workflow but specify that no answer should be generated
        input_message_text = f"Видеосообщение от {telegram_message.from_user.full_name}. Транскрипция звука: {transcription or 'отсутствует'}, аннотация пользователя: {telegram_message.caption or 'отсутствует'}"

        input_message = HumanMessage(input_message_text, additional_kwargs={"no_answer": True})
        self.workflow_controller.invoke_flow(
            {"messages": input_message}, 
            conversation_thread_config
        )


        return transcription