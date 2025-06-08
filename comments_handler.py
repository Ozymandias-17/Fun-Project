async def get_tg_comments(client, username: str|int, limit: int=None, calculate_inf: bool=True):

    import pandas as pd
    from transformers import pipeline
    from tqdm.notebook import tqdm

    clf_emotion = pipeline(task='sentiment-analysis', model='cointegrated/rubert-tiny2-cedr-emotion-detection', top_k=None)
    clf_toxicicity = pipeline(task='sentiment-analysis', model='khvatov/ru_toxicity_detector', top_k=None)
    clf_tonality = pipeline(task='sentiment-analysis', model='seara/rubert-tiny2-russian-sentiment', top_k=None)

    id = []
    FI = []
    user_name = []
    comments_text = []

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


    data = list(zip(id, FI, user_name, comments_text))
    df_comments = pd.DataFrame(data, columns = ["ID", "FI", "Username", "Comment"])

    if calculate_inf == False:
        return pd.DataFrame(data, columns = ["ID", "FI", "Username", "Comment"])

    else:

        prior_emotion = []
        toxicity = []
        neutral_comm = []
        negative_comm = []
        positive_comm = []

        for comm in tqdm(list(df_comments["Comment"])):
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
        df = await get_tg_comments(client, username="username/id", limit=50)
        print(df.head(10))

    asyncio.run(main())