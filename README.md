# Dear Subreddit

The objective of this project is to determine whether a question is more likely to be sourced from the **[AskWomen](https://www.reddit.com/r/AskWomen/)** or **[AskMen](https://www.reddit.com/r/AskMen/)** using various NLP classification models. More specifically, can we create a model that beats the baseline accuracy (50%)? 


## :mag: Background

### About Reddit

Quick background information about **Reddit** from [Wikipedia](https://en.wikipedia.org/wiki/Reddit):

> Reddit is an American social news aggregation, web content rating, and discussion website. Registered members submit content to the site such as links, text posts, images, and videos, which are then voted up or down by other members. Posts are organized by subject into user-created boards called "communities" or "subreddits", which cover a variety of topics such as news, politics, religion, science, movies, video games, music, books, sports, fitness, cooking, pets, and image-sharing.
>

### About AskMen & AskWomen

The AskWomen and AskMen subreddit are used as forums for users to pose questions to a jury of online peers. Questions are mostly associated to social topics, e.g. relationships, friendships, sexuality, etc. Both subreddits do not allow discriminatory langauge and content is moderated. All submissions are in text format (no media, e.g. video or images) and are usually posed in the form of a question.


## :triangular_ruler: Data

For the data collection process, the [Pushshift API](https://github.com/pushshift/api) was used to gather submissions from both subreddits. To facilitate the data calls, the code notebook employs a Pushshift API wrapper library, [pmaw](https://github.com/mattpodolak/pmaw). This helped bypass the 100 submission limit for each API response. For each subreddit, 10,000 submissions were collected, creating a combined dataset containing 20,000 submissions split evenly between the two subreddits.

For data cleaning, HTML links were removed from all text fields (title and self_text). Moreover, text from title and self_text were combined to create one field representing all text.

## :bar_chart: EDA

During the EDA process, I took a look at a two different questions to explore the text data for posts from each subreddit.
- How wordy are the questions?
- What are the popular. words and phrases used within these subreddits?

### 1) How "wordy" are the questions?

Looking at distribution of words per subreddit, the histograms appear to be comprable - one subreddit is not more verbose than the other. On average, questions contain 13 words.

![plot](https://github.com/tashapiro/subreddit-askwomen-askmen/blob/main/plots/dist_all_words.png)

### 2) What are the popular words and phrases used within these subreddits?

Both subreddits have a lot of words and phrases in common - during the analysis of top 15 words and top 15 bigrams for each subreddit respectively, I noticed that 

![plot](https://github.com/tashapiro/subreddit-askwomen-askmen/blob/main/plots/top_words.png)

![plot](https://github.com/tashapiro/subreddit-askwomen-askmen/blob/main/plots/top_bigrams.png)


## :chart_with_upwards_trend: Models

**Overview**

For this project, **Logistic Regression** (LogReg) and **Random Forest Classifier** (RFC) were used in conjunction with NLP, CountVectorizer (CV) and Tfid Vectorizer (TFV). GridSearch was also used to help tune parameters for both models. Data was tested using an 80/20 split for our data set and stratified on the y variable to ensure that the test sample was balanced (50/50 split between AskWomen and AskMen submissions).The X variables incorporated only text data, no additional data points were included.

While all models beat the baseline accuracy (50%), the RFC model yielded the best accuracy (R2) score (78.5%). It is important to note that due to the complexity of the model (introducing more variance), the model is prone to more **overfitting** compared to Logistic Regression: there is almost a 20 percentage point difference between train and test scores vs. the 2 percentage point difference in the Logistic Regression model. 

In all classifier models, AskWomen was codified as the positive value (1). The **sensitivity** and **specificity** scores tell us that all the models are better at correctly classifying AskWomen questions (sensitivity 76-85%) compared to correctly classifying AskMen questions (specifity 70-73%).

**Model Evaluation**

| Model Type       | Train R2 | Test R2 |  TP  |  FP |  TN  |  FN | Precision | Sensitivity | Specificity |
|:-----------------|:--------:|:-------:|:----:|:---:|:----:|:---:|:---------:|:-----------:|:-----------:|
| LogReg (CV)      | 77.7%    | 75.2%   | 1588 | 579 | 1421 | 412 | 73.3%     | 79.4%       | 71.1%       |
| LogReg (TFV)     | 78.5%    | 74.8%   | 1531 | 540 | 1460 | 469 | 73.9%     | 76.6%       | 73.0%       |
| LogReg (CV & GS) | 96.4%    | 76.8%   | 1594 | 523 | 1477 | 406 | 75.3%     | 79.7%       | 73.9%       |
| RFC (CV)         | 99.6%    | 78.4%   | 1694 | 558 | 1442 | 306 | 75.2%     | 84.7%       | 72.1%       |
| RFC (TFV)        | 99.6%    | 77.1%   | 1674 | 589 | 1411 | 326 | 74.0%     | 83.7%       | 70.6%       |
| RFC (CV & GS)    | 99.6%    | 78.5%   | 1707 | 567 | 1433 | 293 | 75.1%     | 85.4%       | 71.7%       |


**Misclassification**

In addition to model evaluation, I looked at examples of misclassified predictions using the tuned RFC model and noticed a common trend: women were also frequenting the AskMen subreddit to solicit advice from their male peers and men were looking for advice on the AskWomen subreddit. Examples of misclassified predictionars are available in the [presentation](https://git.generalassemb.ly/tshapiro/project_3/blob/master/subreddit_presentation.pdf).. 


## :crystal_ball: Streamlit Application
At the end of testing and evaluating models, the best model (RFC CV & GS) was exported as a pickle file. This pickle file was used to build a Streamlit app that allows users to submit their own question and predict whether the question sounds more like a submission from AskMen or AskWomen. Due to file size constraints with Streamlit (pickle file >100MB), the pickle file was hosted on Google Drive.

Try out the Streamlit app [here](https://share.streamlit.io/tashapiro/subreddit-askwomen-askmen/main/code/askmen-askwomen-app.py).


## :memo: Conclusions, Considerations, and Recommendations

- **Themes** – both subreddit forums have posts soliciting advice about sex and relationships.
- **Models** – all models beat 50% baseline accuracy. RFC provides slightly better accuracy, it severely overfits model (goes up in complexity, higher variance).  Model favored single words instead of single words & bigrams, too simple? Test out models with more parameters, try other models
- **Authorship vs. Audience** – NLP classification is good at solving questions of “authorship.” Any gender can post questions on subreddit, e.g. women can post on AskMen subreddit for advice and vice versa.
- **Gender isn’t binary** – aside from overlap between threads, gender isn’t binary (neither is sexuality – important to note given sexual topics). Model doesn’t consider this, operates on a binary assumption.
- **Other demographic variables** – future research should consider introducing other variables, e.g. age, education, and location. These variables may influence both language and theme of posts.
