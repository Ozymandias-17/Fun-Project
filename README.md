# Telegram Posts & Comments Analyzer 

## Overview
This project is a Telegram bot designed to parse and analyze posts and comments from Telegram channels. The bot provides detailed statistics and calculates the tonality and primary emotions of the posts and comments.

## Features
- Parse Channel Posts/Comments: Extract posts and comments from specified Telegram channels;
- Statistics: Provide various statistics about the posts/comments, such as the number of posts, number of comments, users activity, etc;
- Sentiment Analysis: Assess the tonality of the posts/comments (positive, negative, neutral);
- Emotion Analysis: Identify primary emotions in the posts/comments (e.g., happiness, sadness, anger, etc.).

## Libraries Used in Project
- Telegram API: Telethon, TeleBot;
- Data Analysis: Pandas, Matplotlib
- Sentiment Analysis: Transformers

## Commands
- /start: Greeting user. Show main commands;
- /group_posts: Parse channel posts and decompose them by toxicity, tonality and emotions;
- /group_comments: Parse channel comments and decompose them by toxicity, tonality and emotions; statistic for most active commentators.
