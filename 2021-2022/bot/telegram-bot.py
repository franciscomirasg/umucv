from telegram.ext import Updater, CommandHandler
from telegram import Update, Bot
import subprocess
from os import environ

MI_ID = environ.get('USER_ID')

UPDATER = Updater(environ.get('TOKEN'), use_context=True)
dispatcher = UPDATER.dispatcher
bot: Bot = UPDATER.bot


def get_ID(update: Update, context):
    context.bot.send_message(chat_id=update.effective_chat.id,
                             text=f'Chat ID:`{update.effective_chat.id}`\nUser ID: `{update.effective_user}`')


start_handler = CommandHandler('getID', get_ID)
dispatcher.add_handler(start_handler)


def shutdown():
    UPDATER.stop()
    UPDATER.is_idle = False
    # cam.stop()


start_handler = CommandHandler('shutdown', shutdown)
dispatcher.add_handler(start_handler)

UPDATER.start_polling()
bot.send_message(MI_ID, f'Hey, I am Online')
