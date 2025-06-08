from telethon import TelegramClient
from telebot import async_telebot

import asyncio
import nest_asyncio

import pandas as pd

from dotenv import load_dotenv
import os

from posts_handler import get_tg_channel
from comments_handler import get_tg_comments
from visualization import views_distribution, toxic_and_emotion_stat, top_commentators

nest_asyncio.apply()
load_dotenv()

api_id = os.getenv("API_ID") 
api_hash = os.getenv("API_HASH") 
bot_token = os.getenv("BOT_TOKEN") 

client = TelegramClient("bot", api_id, api_hash)   # Here "bot" — is the name of the file that will be created to store application authorization data
bot = async_telebot.AsyncTeleBot(bot_token)


# commands
post_info = "group_posts"
comments_info = "group_comments"
last_command = []

@bot.message_handler(commands=['start'])
async def handle_start(message):
    # Отправка сообщения: приветствие
    if message.from_user.username != None:
        await bot.send_message(message.chat.id,
                               (f'Здравствуй, {message.from_user.username}! '
                                'Я могу спарсить телеграм канал и сделать некоторую статистическую сводку по постам и комментариям данного канала.')
                                )
    else:
        await bot.send_message(message.chat.id,
                               (f'Здравствуй, {message.from_user.first_name}! '
                               'Я могу спарсить телеграм канал и сделать некоторую статистическую сводку по постам и комментариям данного канала.')
                               )

@bot.message_handler(commands=[post_info, comments_info])
async def handle_message(message):
    await bot.reply_to(message, 'Введи id или username группы, по которой хочешь получить информацию, и limit (необязательно), через запятую: mygroup, 500')
    last_command.append(message.text)


@bot.message_handler(func=lambda message:True)
async def handle_message(message):
    
    try:
        parts = [x.strip() for x in message.text.split(",")]
        response = int(parts[0]) if parts[0].isdigit() else parts[0]
        limit = int(parts[1]) if len(parts) > 1 else 1000
    
    except:
        await bot.send_message(message.chat.id, "Формат неправильный. Введите: username или id, optional limit")

    if last_command[-1] == ("/" + post_info):

        df_bot = await get_tg_channel(client, response, limit=limit)

        df_bot.to_html("Data_with_posts.html")

        views_distribution(df_bot)
        toxic_and_emotion_stat(df_bot, "Posts")

        await bot.send_document(message.chat.id, document=open("Data_with_posts.html", "rb"))
        await bot.send_photo(message.chat.id, photo=open("Views_distribution.png", "rb"))
        await bot.send_photo(message.chat.id, photo=open("Result.png", "rb"))

        last_command.append(0)

    elif last_command[-1] == ("/" + comments_info):

        try:
            await bot.send_message(message.chat.id, 'Придётся немного подождать, идёт сбор комментариев...')

            global df_comments_bot
            df_comments_bot = await get_tg_comments(client, response, limit=limit)

            df_comments_bot.to_html("Data_with_comments.html")

            toxic_and_emotion_stat(df_comments_bot, "Comments")
            top_commentators(df_comments_bot)

            await bot.send_document(message.chat.id, document=open("Data_with_comments.html", "rb"))
            await bot.send_photo(message.chat.id, photo=open("Result.png", "rb"))
            await bot.send_photo(message.chat.id, photo=open("Top_commentators.png", "rb"))
            await bot.send_message(message.chat.id, ('Если тебя интересуют конкретные пользователи (пользователь), ты можешь прислать мне их username. '
                                                     'Я предоставлю сводку по комментариям этих пользовотелей в данной группе.'))

        except:
            await bot.send_message(message.chat.id, 'Перепроверь свой запрос, я не могу его обработать')

        last_command.append(0)

    elif set(response.split(sep=", ")).issubset(df_comments_bot["Username"].unique()):

        wanted_users = set(response.split(sep=", ")).intersection(set(df_comments_bot["Username"].unique()))
        filter_df_comments = df_comments_bot[df_comments_bot["Username"].isin(wanted_users)]
        tonality_of_spec_users = filter_df_comments.groupby("Username")[["Neutral", "Negative", "Positive"]].agg(['sum', 'mean'])
        emotion_of_spec_users = pd.DataFrame(filter_df_comments.groupby("Username")["Prior Emotion"].value_counts())
        toxic_of_spec_users = pd.DataFrame(filter_df_comments.groupby("Username")["Toxicity"].value_counts())

        filter_df_comments.to_html("Data_with_spec_users_comments.html")
        tonality_of_spec_users.to_html("Tonality_spec_users_comments.html")
        emotion_of_spec_users.to_html("Emotion_spec_users_comments.html")
        toxic_of_spec_users.to_html("Toxic_spec_users_comments.html")

        await bot.send_document(message.chat.id, document=open("Data_with_spec_users_comments.html", "rb"))
        await bot.send_document(message.chat.id, document=open("Tonality_spec_users_comments.html", "rb"))
        await bot.send_document(message.chat.id, document=open("Emotion_spec_users_comments.html", "rb"))
        await bot.send_document(message.chat.id, document=open("Toxic_spec_users_comments.html", "rb"))

    else:
        await bot.send_message(message.chat.id, 'Перепроверь свой запрос, я не могу его обработать')


async def main():
    await client.start()  
    await bot.polling()   

asyncio.run(main())