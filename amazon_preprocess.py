import gzip
import pickle
import numpy as np
import scipy.sparse as sp
import random as rd
import math
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from transformers import DistilBertTokenizer, DistilBertModel

# import xgboost as xgb

# from run_ours import pos_neg_split, undersample

rd.seed(1)


def parse(path):
    g = gzip.open(path, "rb")
    for l in g:
        yield eval(l)


def getdict(path):
    reviews = {}
    for d in parse(path):
        review = list(d.values())
        if review[0] not in reviews.keys():
            reviews[review[0]] = []
        reviews[review[0]].append(d)
    return reviews


def date_diff(t1, t2):
    date1 = datetime.fromtimestamp(t1)
    date2 = datetime.fromtimestamp(t2)

    return abs((date1 - date2).days)


def has_overlapping_star(reviews1, reviews2):
    # reviews: [(star, time), ...)]
    flag = False
    for r1 in reviews1:
        for r2 in reviews2:
            if r1[0] == r2[0]:
                flag = True
                break
    return flag


def has_similar_time(reviews1, reviews2, diff=7):
    flag = False
    for r1 in reviews1:
        for r2 in reviews2:
            if date_diff(r1[1], r2[1]) <= diff:
                flag = True
                break
    return flag


def time_entropy(timestamps):
    dates = [datetime.fromtimestamp(stamp) for stamp in timestamps]

    unique_years = list(set([date.year for date in dates]))

    no_date = {year: 0 for year in unique_years}

    for date in dates:
        no_date[date.year] += 1

    date_ratio = [d / len(dates) for d in no_date.values()]

    entropy = -sum([ratio * math.log(ratio) for ratio in date_ratio])

    return entropy


def cal_rating(new_reviews):
    no_rating = sp.lil_matrix((len(new_reviews), 17))

    for i, u in enumerate(new_reviews):
        ratings = [int(review["overall"]) for review in new_reviews[u]]
        # feature [2-11]
        vector = [ratings.count(r + 1) for r in range(5)] + [ratings.count(r + 1) / len(ratings) for r in range(5)]
        # feature [12-13]
        vector += [vector[5] + vector[6]] + [vector[8] + vector[9]]
        # feature [14]
        entropy = -sum([ratio * math.log(ratio + 1e-5) for ratio in vector[5:10]])
        vector += [entropy]
        # feature [15-18]
        vector += [np.median(ratings)] + [max(ratings)] + [min(ratings)] + [np.mean(ratings)]

        no_rating[i, :] = np.reshape(vector, (1, 17))

    return no_rating


def cal_votes(new_reviews):
    no_votes = sp.lil_matrix((len(new_reviews), 12))

    for i, u in enumerate(new_reviews):
        votes = [review["helpful"] for review in new_reviews[u]]
        help = [vote[0] for vote in votes]
        total = [vote[1] for vote in votes]
        unhelp = [vote[1] - vote[0] for vote in votes]
        # feature [19-20]
        vector = [sum(help)] + [sum(unhelp)]
        # feature [21-24]
        vector += (
            [sum(help) / (sum(total) + 1e-5)]
            + [sum(help) / len(votes)]
            + [sum(unhelp) / (sum(total) + 1e-5)]
            + [sum(unhelp) / len(votes)]
        )
        # feature [25-30]
        vector += [np.median(help)] + [max(help)] + [min(help)] + [np.median(unhelp)] + [max(unhelp)] + [min(unhelp)]

        no_votes[i, :] = np.reshape(vector, (1, 12))

    return no_votes


def cla_dates(new_reviews):
    no_dates = sp.lil_matrix((len(new_reviews), 3))

    for i, u in enumerate(new_reviews):
        timestamps = [review["unixReviewTime"] for review in new_reviews[u]]
        duration = date_diff(max(timestamps), min(timestamps))
        # feature [31]
        vector = [duration]
        # feature [32]
        vector += [time_entropy(timestamps)]
        # feature [33]
        vector += [1] if duration == 0 else [0]

        no_dates[i, :] = np.reshape(vector, (1, 3))

    return no_dates


def nltk_sentiment(sentence):
    nltk_sentiment = SentimentIntensityAnalyzer()
    score = nltk_sentiment.polarity_scores(sentence)
    return score


def sentiment(new_reviews):
    user_texts = [" ".join([review["reviewText"] for review in reviews]) for reviews in new_reviews.values()]

    nltk_results = []
    for i, text in enumerate(user_texts):
        print(i)
        result = nltk_sentiment(text)
        if result["compound"] > 0:
            nltk_results.append(1)
        elif result["compound"] < 0:
            nltk_results.append(-1)
        else:
            nltk_results.append(0)

    return nltk_results


def build_graph(new_reviews, type):
    user_adj = sp.lil_matrix((len(new_reviews), len(new_reviews)))

    # user-product-user
    if type == "upu":
        products = {u: [review["asin"] for review in reviews] for u, reviews in new_reviews.items()}
        for i1, u1 in enumerate(new_reviews):
            # print(i1)
            for i2, u2 in enumerate(new_reviews):
                if u1 != u2 and len(list(set(products[u1]) & set(products[u2]))) >= 1:
                    user_adj[i1, i2] = 1
                    user_adj[i2, i1] = 1

    # user-star&time-user
    elif type == "usu":
        rating_time = {
            user: [(review["overall"], review["unixReviewTime"]) for review in reviews]
            for user, reviews in new_reviews.items()
        }

        count = 0
        new_reviews_keys = list(new_reviews.keys())
        for i1, u1 in enumerate(new_reviews_keys):
            print(i1)
            for i2, u2 in enumerate(new_reviews_keys[i1 + 1 :]):
                if (
                    u1 != u2
                    and has_overlapping_star(rating_time[u1], rating_time[u2])
                    and has_similar_time(rating_time[u1], rating_time[u2])
                ):
                    count += 1
                    user_adj[i1, i2] = 1
                    user_adj[i2, i1] = 1
        print("Count: ", count)

    # user-textsim_user
    elif type == "uvu":
        user_text = {
            user: " ".join([review["reviewText"] for review in reviews]) for user, reviews in new_reviews.items()
        }

        all_text = list(user_text.values())
        vect = TfidfVectorizer(min_df=1, stop_words="english")
        tfidf = vect.fit_transform(all_text)
        simi = tfidf * tfidf.T
        simi_arr = simi.toarray()
        np.fill_diagonal(simi_arr, 0)
        flat_arr = simi_arr.flatten()
        flat_arr.sort()
        threshold = flat_arr[int(flat_arr.size * 0.95)]
        print(threshold)

        for i1, u1 in enumerate(new_reviews):
            print(i1)
            for i2, u2 in enumerate(new_reviews):
                if u1 != u2 and simi_arr[i1, i2] >= threshold:
                    user_adj[i1, i2] = 1
                    user_adj[i2, i1] = 1

        # user - product - star - user
        # products = {u: [review["asin"] for review in reviews] for u, reviews in new_reviews.items()}
        # rating_time = {
        #     user: [(review["overall"], review["unixReviewTime"]) for review in reviews]
        #     for user, reviews in new_reviews.items()
        # }
        # for i1, u1 in enumerate(new_reviews):
        #     print(i1)
        #     for i2, u2 in enumerate(new_reviews):
        #         if (
        #             u1 != u2
        #             and len(list(set(products[u1]) & set(products[u2]))) >= 1
        #             and has_overlapping_star(rating_time[u1], rating_time[u2])
        #         ):
        #             user_adj[i1, i2] = 1
        #             user_adj[i2, i1] = 1

    elif type == "ubu":
        user_text = {
            user: [review["reviewText"] for review in reviews] for user, reviews in new_reviews.items()
        }

        model = DistilBertModel.from_pretrained("distilbert-base-cased").to("cuda")
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-cased")

        def encode(review_texts: list[str]):
            vectors = []
            for text in review_texts:
                inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to("cuda")
                outputs = model(**inputs)
                vectors.append(outputs[0][0, 0].detach().cpu().numpy())

            vector = np.mean(vectors, axis=0)
            # L2 normalization
            vector = vector / np.linalg.norm(vector)
            
            return vector

        user_bert = {
            user: encode(reviews) for user, reviews in user_text.items()
        }

        new_reviews_keys = list(new_reviews.keys())

        # Similar if cossim > 0.8
        for i1, u1 in enumerate(new_reviews_keys):
            print(i1)
            for i2, u2 in enumerate(new_reviews_keys[i1 + 1 :]):
                if u1 != u2 and np.dot(user_bert[u1], user_bert[u2]) > 0.8:
                    user_adj[i1, i2] = 1
                    user_adj[i2, i1] = 1
 
    else:
        raise ValueError("Invalid type")

    return user_adj


def build_features(new_reviews):
    features = sp.lil_matrix((len(new_reviews), 36))

    user_ids = list(new_reviews.keys())

    # 1) [0] Number of rated products
    no_prod = [len(reviews) for reviews in new_reviews.values()]
    features[:, 0] = np.reshape(no_prod, (len(no_prod), 1))

    # 2) [1] Length of username
    len_name = [
        len(reviews[0]["reviewerName"]) if "reviewerName" in reviews[0].keys() else 0
        for reviews in new_reviews.values()
    ]
    features[:, 1] = np.reshape(len_name, (len(len_name), 1))

    # 3) [2-11] Number and ratio of each rating level given by a user
    # 4) [12-13] Ratio of positive and negative ratings (4,5)-pos, (1,2)-neg
    # 5) [14] Entropy of ratings -\sum_{r}(percentage_r * \log percentage_{r})
    # 6) [15-18] Median, min, max, and average of ratings
    no_rating = cal_rating(new_reviews)
    features[:, range(2, 19)] = no_rating

    # 7) [19-20] Total number of helpful and unhelpful votes a user gets
    # 8) [21-24] The ratio and mean of helpful and unhelpful votes
    # 9) [25-30] Median, min, and max number of helpful and unhelpful votes
    no_votes = cal_votes(new_reviews)
    features[:, range(19, 31)] = no_votes

    # 10) [31] Day gap
    # 11) [32] Time entropy
    # 12) [33] Same date indicator
    no_dates = cla_dates(new_reviews)
    features[:, range(31, 34)] = no_dates

    # 13) [34] Feedback summary length
    summ_length = [
        sum([len(review["summary"]) if "summary" in reviews[0].keys() else 0 for review in reviews]) / len(reviews)
        for reviews in new_reviews.values()
    ]
    features[:, 34] = np.reshape(summ_length, (len(summ_length), 1))

    # 14) [35] Review text sentiment
    senti_scores = sentiment(new_reviews)
    features[:, 35] = np.reshape(senti_scores, (len(senti_scores), 1))

    return features


if __name__ == "__main__":
    try:
        reviews = pickle.load(open("musical_reviews.pickle", "rb"))
    except:
        reviews = getdict("reviews_Musical_Instruments.json.gz")
        pickle.dump(reviews, open("musical_reviews.pickle", "wb"))

    all_reviews = reviews

    # create ground truth
    user_labels = []
    labeled_reviews = {}
    for u, total in all_reviews.items():
        # print([single[2][0] for single in total])
        # print([single[2][1] for single in total])
        helpful = sum([single["helpful"][0] for single in total])
        votes = sum([single["helpful"][1] for single in total])

        if votes >= 20:
            if helpful / votes > 0.8:
                labeled_reviews[u] = total
                user_labels.append(0)
            elif helpful / votes < 0.2:
                labeled_reviews[u] = total
                user_labels.append(1)

    all_user = set(list(all_reviews.keys()))
    label_user = set(list(labeled_reviews.keys()))
    unlabel_user = list(all_user - label_user)

    sampled_users = rd.sample(unlabel_user, int(len(unlabel_user) * 0.01))

    sampled_reviews = {user: all_reviews[user] for user in sampled_users}

    new_reviews = labeled_reviews.copy()
    for u, t in sampled_reviews.items():
        if u in new_reviews:
            continue
            
        new_reviews[u] = t
        user_labels.append(-100)

    user_labels = np.array(user_labels)
    print(len(user_labels))
    with open("amazon_instruments_labels.pkl", "wb") as f:
        pickle.dump(user_labels, f)

    sp.save_npz("amazon_instruments_user_product_user", build_graph(new_reviews, "upu").tocsr())
    sp.save_npz("amazon_instruments_user_star_time_user", build_graph(new_reviews, "usu").tocsr())
    sp.save_npz("amazon_instruments_user_tfidf_user", build_graph(new_reviews, "uvu").tocsr())
    sp.save_npz("amazon_instruments_user_bert_user.npz", build_graph(new_reviews, "ubu").tocsr())

    # construct feature vectors
    features = build_features(new_reviews)
    sp.save_npz("amazon_instruments_features.npz", features.tocsr())
