# Download libraries
!pip install telethon==1.34
!pip install pyTelegramBotAPI==4.21
from telethon import TelegramClient
from telebot import TeleBot, types, async_telebot
import asyncio
import nest_asyncio
from tqdm.notebook import tqdm
import numpy as np
import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt
nest_asyncio.apply()


# Authorization
api_id = "YOUR_ID"
api_hash = "YOUR_HASH"
bot_token = "YOUR_BOT_TOKEN"
client = TelegramClient("test", api_id, api_hash) # Here "test" — is the name of the file that will be created to store application authorization data
client.connect() # or .start()


# Function for parsing channel posts
async def get_tg_channel(username, limit=None):    
    username = await client.get_entity(username)
    messages = client.iter_messages(username, limit=limit)
    
    async for message in messages:
        if message.text != None and len(message.text) > 1:
            post_id.append(message.id)
            post_text.append(message.text.replace("\n", " ").replace("  ", " "))
            post_dates.append(message.date)
            post_views.append(message.views)
            
            try:
                post_comments.append(message.replies.replies)
            except:
                post_comments.append(0)

            try:
                count_reac = []
                reac_types = []
                for reac in message.reactions.results:
                    try:
                        count_reac.append(reac.count)
                        reac_types.append(reac.reaction.emoticon)
                    except:
                        continue

                post_likes.append(sum(count_reac))
                post_emotion_types.append(reac_types)

            except:
                post_likes.append(0)
                post_emotion_types.append("-")

# Function to make dataframe with posts
def make_dataframe_posts(calculate_inf=True):
    response_1 = np.array(post_likes)/np.array(post_views)
    response_2 = np.array(post_comments)/np.array(post_views)

    data = list(zip(post_id, post_dates, post_text, post_views, post_likes,
                    post_emotion_types, post_comments, response_1, response_2))

    df = pd.DataFrame(data, columns = ["ID", "Date", "Text", "Views_Count", "Reaction_Count", "Emotion_Types",
                                       "Comments_Count", "Response_1", "Response_2"])

    if calculate_inf:
        clf_emotion = pipeline(task='sentiment-analysis', model='cointegrated/rubert-tiny2-cedr-emotion-detection', top_k=None)
        clf_toxicicity = pipeline(task='sentiment-analysis', model='khvatov/ru_toxicity_detector', top_k=None)
        clf_tonality = pipeline(task='sentiment-analysis', model='seara/rubert-tiny2-russian-sentiment', top_k=None)

        prior_emotion = []
        toxicity = []
        neutral = []
        negative = []
        positive = []

        for t in tqdm(df["Text"]):
            emotion_analyse = clf_emotion(t)
            tox_analyse = clf_toxicicity(t)
            ton_analyse = clf_tonality(t)
            prior_emotion.append(emotion_analyse[0][0]["label"])
            if tox_analyse[0][0]["label"] == "LABEL_0":
                toxicity.append("non toxic")
            else:
                toxicity.append("toxic")

            for row_ton in ton_analyse[0]:
                if row_ton['label'].lower() == 'neutral':
                    neutral.append(row_ton['score'])
                elif row_ton['label'].lower() == 'negative':
                    negative.append(row_ton['score'])
                elif row_ton['label'].lower() == 'positive':
                    positive.append(row_ton['score'])

        calculated_inf = pd.DataFrame(list(zip(prior_emotion, toxicity, neutral, negative, positive)),
                                      columns=["Prior Emotion", "Toxicity", "Neutral", "Negative", "Positive"])

        return df.join(calculated_inf).set_index("Date")

    else:
        return df.set_index("Date")


# Function for parsing channel comments
async def get_tg_comments(username, limit=None):    
    username = await client.get_entity(username)
    messages = client.iter_messages(username, limit=limit)
    
    async for message in messages:
        try:
            async for reply in client.iter_messages(username, reply_to=message.id):
                if len(reply.message) > 1:
                    try:
                        id.append(reply.from_id.user_id)
                        us = await client.get_entity(reply.from_id.user_id)
                        FI.append(" ".join([str(us.first_name), str(us.last_name)]).replace("None", "-"))
                        user_name.append(str(us.username).replace("None", "-"))
                    except:
                        id.append(reply.from_id.channel_id)
                        us = await client.get_entity(reply.from_id.channel_id)
                        FI.append("Admin")
                        user_name.append(str(us.title))
                    comments_text.append(reply.message.replace("\n", " ").replace("  ", " "))
        
        except:
            continue

# Function to make dataframe with comments
def make_dataframe_comments(calculate_inf=True):
    data = list(zip(id, FI, user_name, comments_text))
    df_comments = pd.DataFrame(data, columns = ["ID", "FI", "Username", "Comment"])

    if calculate_inf == False:
        clf_emotion = pipeline(task='sentiment-analysis', model='cointegrated/rubert-tiny2-cedr-emotion-detection', top_k=None)
        clf_toxicicity = pipeline(task='sentiment-analysis', model='khvatov/ru_toxicity_detector', top_k=None)
        clf_tonality = pipeline(task='sentiment-analysis', model='seara/rubert-tiny2-russian-sentiment', top_k=None)

        prior_emotion = []
        toxicity = []
        neutral_comm = []
        negative_comm = []
        positive_comm = []

        for comm in tqdm(df_comments["Comment"]):
            emotion_analyse = clf_emotion(comm)
            tox_analyse = clf_toxicicity(comm)
            ton_analyse = clf_tonality(comm)
            prior_emotion.append(emotion_analyse[0][0]["label"])
            if tox_analyse[0][0]["label"] == "LABEL_0":
                toxicity.append("non toxic")
            else:
                toxicity.append("toxic")

            for row_ton in ton_analyse[0]:
                if row_ton['label'].lower() == 'neutral':
                    neutral_comm.append(row_ton['score'])
                elif row_ton['label'].lower() == 'negative':
                    negative_comm.append(row_ton['score'])
                elif row_ton['label'].lower() == 'positive':
                    positive_comm.append(row_ton['score'])

        calculated_inf = pd.DataFrame(list(zip(prior_emotion, toxicity, neutral_comm, negative_comm, positive_comm)),
                                      columns=["Prior Emotion", "Toxicity", "Neutral", "Negative", "Positive"])

        return df_comments.join(calculated_inf)

    else:
        return pd.DataFrame(data, columns = ["ID", "FI", "Username", "Comment"])


# Distribution of views
def views_distribution(df):
    plt.style.use("bmh")
    plt.figure(figsize=(10, 5), constrained_layout=True)
    plt.bar(df.index, df["Views_Count"], color="#6C92AF")
    plt.title("Distribution of views", fontsize=16)
    plt.gca().set_axisbelow(True)
    plt.savefig("Views_distribution.png")

# Toxicity, emotions and tonality overview
def toxic_and_emotion_stat(df, object):
    plt.style.use("bmh")
    fig, axs = plt.subplots(3, figsize=(6, 6), constrained_layout=True)
    axs[0].bar(df["Toxicity"].value_counts().index, df["Toxicity"].value_counts(), color="teal", width=0.3)
    axs[0].set_title(f"Toxicity of {object}", fontsize=13)
    axs[0].set_ylabel("Frequency")
    axs[0].set_axisbelow(True)
    axs[1].bar(df["Prior Emotion"].value_counts().index, df["Prior Emotion"].value_counts(), color="palevioletred")
    axs[1].set_title(f"Emotions of {object}", fontsize=13)
    axs[1].set_ylabel("Frequency")
    axs[1].set_axisbelow(True)
    axs[2].bar(df[["Neutral", "Negative", "Positive"]].mean().index, df[["Neutral", "Negative", "Positive"]].mean(),
        color="#2A6478", width=0.4)
    axs[2].set_title(f"Tonality of {object}", fontsize=13)
    axs[2].set_ylabel("Mean Coefficient")
    axs[2].set_axisbelow(True)
    plt.savefig("Result.png")

# Most active commentators
def top_commentators(data, how_many=15):
    top_comm = data["Username"].value_counts()[0:how_many]
    plt.style.use("bmh")
    plt.figure(figsize=(12, 6), constrained_layout=True)
    plt.bar(top_comm.index, top_comm, color="goldenrod")
    plt.xticks(rotation=55, fontsize=11)
    plt.yticks(fontsize=11)
    plt.ylabel("Number of Comments")
    plt.title("Most active commentators", fontsize=16)
    plt.gca().set_axisbelow(True)
    plt.savefig("Top_commentators.png")


# Creating Bot
bot = async_telebot.AsyncTeleBot(bot_token)

# commands
post_info = "group_posts"
comments_info = "group_comments"
comand = []

@bot.message_handler(commands=['start'])
async def handle_start(message):
    # Add keys for commands
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    item1 = types.KeyboardButton("/" + post_info)
    item2 = types.KeyboardButton("/" + comments_info)
    markup.add(item1, item2)

    if message.from_user.username:
        await bot.send_message(message.chat.id,
                               f'Здравствуй, {message.from_user.username}! Я могу спарсить телеграм канал и сделать некоторую статистическую сводку по постам и комментариям данного канала. Чтобы начать посмотри меню основных команд или нажми одну из кнопок внизу.', reply_markup=markup)

    else:
        await bot.send_message(message.chat.id,
                               f'Здравствуй, {message.from_user.first_name}! Я могу спарсить телеграм канал и сделать некоторую статистическую сводку по постам и комментариям данного канала. Чтобы начать посмотри меню основных команд или нажми одну из кнопок внизу.', reply_markup=markup)

@bot.message_handler(commands=[post_info, comments_info])
async def handle_message(message):
    await bot.reply_to(message, 'Введи id или username группы, по которой хочешь получить информацию')
    comand.append(message.text)

post_id = []
post_dates = []
post_text = []
post_likes = []
post_emotion_types = []
post_views = []
post_comments = []

id = []
FI = []
user_name = []
comments_text = []

@bot.message_handler(func=lambda message:True)
async def handle_message(message):
    global response

    try:
        response = int(message.text)
    except:
        response = message.text

    if comand[-1] == ("/" + post_info):

        try:
            await bot.send_message(message.chat.id, 'Потребуется немного времени, чтобы собрать все посты в группе...')
    
            async with client:
                client.loop.run_until_complete(get_tg_channel(response))
    
            df_bot = make_dataframe_posts()
            df_bot.to_html("Data_with_posts.html")
            views_distribution(df_bot)
            toxic_and_emotion_stat(df_bot, "Posts")
    
            await bot.send_document(message.chat.id, document=open("Data_with_posts.html", "rb"))
            await bot.send_photo(message.chat.id, photo=open("Views_distribution.png", "rb"))
            await bot.send_photo(message.chat.id, photo=open("Result.png", "rb"))

        except:
            await bot.send_message(message.chat.id, 'Перепроверь свой запрос, я не могу его обработать')        
        
        post_id.clear()
        post_dates.clear()
        post_text.clear()
        post_likes.clear()
        post_emotion_types.clear()
        post_views.clear()
        post_comments.clear()
        comand.append(0)

    elif comand[-1] == ("/" + comments_info):
        
        try:
            await bot.send_message(message.chat.id, 'Придётся немного подождать, идёт сбор комментариев...')
    
            async with client:
                client.loop.run_until_complete(get_tg_comments(response, limit=100))
    
            global df_comments_bot
            df_comments_bot = make_dataframe_comments()
            df_comments_bot.to_html("Data_with_comments.html")
            toxic_and_emotion_stat(df_comments_bot, "Comments")
            top_commentators(df_comments_bot)
    
            await bot.send_document(message.chat.id, document=open("Data_with_comments.html", "rb"))
            await bot.send_photo(message.chat.id, photo=open("Result.png", "rb"))
            await bot.send_photo(message.chat.id, photo=open("Top_commentators.png", "rb"))
            await bot.send_message(message.chat.id, '''Если тебя интересуют конкретные пользователи (пользователь), ты можешь прислать мне их username.
                                                       Я предоставлю сводку по комментариям этих пользовотелей в данной группе.''')
        
        except:
            await bot.send_message(message.chat.id, 'Перепроверь свой запрос, я не могу его обработать')
    
        id.clear()
        FI.clear()
        user_name.clear()
        comments_text.clear()
        comand.append(0)

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


asyncio.run(bot.polling(none_stop=True))
