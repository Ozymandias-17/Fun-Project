async def get_tg_channel(client, username: str|int, limit: int=None, get_users: bool=False, calculate_inf: bool=True):

    import numpy as np
    import pandas as pd
    from transformers import pipeline
    from tqdm.notebook import tqdm

    clf_emotion = pipeline(task='sentiment-analysis', model='cointegrated/rubert-tiny2-cedr-emotion-detection', top_k=None)
    clf_toxicicity = pipeline(task='sentiment-analysis', model='khvatov/ru_toxicity_detector', top_k=None)
    clf_tonality = pipeline(task='sentiment-analysis', model='seara/rubert-tiny2-russian-sentiment', top_k=None)

    post_id = []
    post_dates = []
    post_text = []
    post_likes = []
    post_emotion_types = []
    post_views = []
    post_comments = []
    users_list = []

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

    if get_users == True:
        users = client.iter_participants(username)
        async for user in users:
            users_list.append(user.username)
    

    response_1 = np.array(post_likes) / np.array(post_views)
    response_2 = np.array(post_comments) / np.array(post_views)

    data = list(zip(post_id, post_dates, post_text, post_views, post_likes,
                    post_emotion_types, post_comments, response_1, response_2))

    df = pd.DataFrame(data, columns = ["ID", "Date", "Text", "Views_Count", "Reaction_Count", "Emotion_Types",
                                       "Comments_Count", "Response_1", "Response_2"])

    if calculate_inf==False:
        return df

    else:

        prior_emotion = []
        toxicity = []
        neutral = []
        negative = []
        positive = []

        for t in tqdm(list(df["Text"])):

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
        
        return df.join(calculated_inf)
    


if __name__ == "__main__":
    import asyncio
    from telethon import TelegramClient
    import nest_asyncio
    from dotenv import load_dotenv
    import os
    nest_asyncio.apply()
    load_dotenv()

    api_id = os.getenv("API_ID") 
    api_hash = os.getenv("API_HASH") 
    client = TelegramClient("bot", api_id, api_hash) 

    async def main():
        await client.start()
        df = await get_tg_channel(client, username="username/id", limit=100)
        print(df.head(10))

    asyncio.run(main())