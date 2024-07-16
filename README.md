# Telegram Posts & Comments Analyzer 

## Overview
This project is a Telegram bot designed to parse and analyze posts and comments from Telegram channels. The bot provides detailed statistics and calculates the toxicity, tonality and primary emotions of the posts and comments.

## Features
- Parse Channel Posts/Comments: Extract posts and comments from specified Telegram channels;
- Statistics: Provide various statistics about the posts/comments, such as distribution of views, users activity, etc;
- Toxicity Analysis: Categorize the toxicity of the posts/comments (toxic, non toxic);
- Sentiment Analysis: Assess the tonality of the posts/comments (positive, negative, neutral);
- Emotion Analysis: Identify primary emotions in the posts/comments (e.g., joy, sadness, anger, etc.).

## Libraries Used in Project
- Telegram API: Telethon, TeleBot;
- Data Analysis: Pandas, Matplotlib
- Sentiment Analysis: Transformers (models from Hugging Face)

## Commands
- /start: Greeting user. Show main commands;
- /group_posts: Parse channel posts and decompose them by toxicity, tonality and emotions;
- /group_comments: Parse channel comments and decompose them by toxicity, tonality and emotions. Statistic for most active commentators.
