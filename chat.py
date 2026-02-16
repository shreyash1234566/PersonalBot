"""
Chat — Interactive CLI
======================
Start an interactive chat session with Ayush's digital twin.

Usage: python chat.py
"""

from src.chatbot import Chatbot


def main():
    bot = Chatbot()
    bot.set_conversation("default", "a girl")

    try:
        bot.chat_cli()
    finally:
        bot.close()


if __name__ == "__main__":
    main()
