# Import the libraries

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from elasticsearch import Elasticsearch
from time import sleep
import pandas as pd
from sklearn.svm import LinearSVC

import sys

# Construct ES / Connect to Elasticsearch cluster 
es = Elasticsearch(
    ['http://127.0.0.1:9200'],
    basic_auth=('username', 'password')
)

# Continously loop to collect and apply logic at interval
while True:
    try:
        get_es_index = es.search(index=elastic_index_model, body={
        "size": 10000,
        "query": {
            "bool": {
            "filter": [
                {
                "range": {
                    "timestamp": {
                    "gte": "now-12M/m",
                    "lte": "now/m"
                    }
                }
                }
            ],
            "must": [
                {
                "exists": {
                    "field": "pnl"
                }
                },
                {
                "exists": {
                    "field": "MACD_DAY_CURVED"
                }
                },
                {
                "match": {
                    "side.keyword": "Buy"
                }
                }
            ]
            }
        }
        })
    except Exception as e:
        print(str(e))
        while True:
            print("Unable to connect to Elasticsearch")
            sleep(60)


    res = get_es_index["hits"]["hits"]

    print(get_es_index["hits"]["total"]["value"])


    # Extracting "_source" field and normalizing it into a DataFrame
    gogo = pd.json_normalize([entry["_source"] for entry in res])

    # Select the features you'd like to work on out of your dataset
    df_live = gogo [["symbol.keyword","ABOVE","ABOVE_1H","ABOVE_1H_BIG","ABOVE_1H_STAMP","btc_24h_now","BUY_H_NEGATIVE","BUY_H_POSITIVE","cmc_rank","condition","eth_24h_now","EXTREME_ABOVE","INDICATOR_1","INDICATOR_2","INDICATOR_3","INDICATOR_4","INDICATOR_5","INDICATOR_6","INDICATOR_7","INDICATOR_8","INDICATOR_9","INDICATOR_10","INDICATOR_11","JUST_CROSS_ABOVE","macd","macd_sline","macd_btc","macd_btc_sline","MACD_DAY_CURVED","MACD_DAY_NEGATIVE","MACD_DAY_POSITIVE","MACD_DAY_POSITIVE_BY_2","MACD_DAY_POSITIVE_EXTREME","market_count_15m","market_count_30m","market_count_now","outcome","pct.24h","pct.24h_8d.avg","percentiles_1","percentiles_5","percentiles_25","percentiles_50","percentiles_75","percentiles_95","percentiles_99","pnl","price_diff_change","price_diff_change_10h","price_diff_change_1d","ROC","roc_9d","roc_9d_btc","ROC_CURVED","ROC_CURVED_DAY_NEGATIVE","ROC_CURVED_DAY_POSITIVE","ROC_CURVED_HOUR_NEGATIVE","ROC_CURVED_HOUR_POSITIVE","ROC_CURVED_NOW","ROC_H","rsi","rsi_btc","RSI_CURVED","RSI_CURVED_H","RSI_DOWN_H","RSI_H","RSI_UP","STILL_ABOVE","tendancies_pct_down","tendancies_pct_up","track_candle_down_avg","track_candle_down_count","track_candle_up_avg","track_candle_up_count","volume_bybit_90d","volume_bybit_now","volume.90d.avg","volume.now","WHOLE_MARKET_CURVED","worked"]]

    # Make a copy of the original dataset so we can work with it (feature engineering)
    df = pd.DataFrame(df_live).copy()

    # Convert booleans to numerical 1 or 0
    booleans_to_numeric = ["ABOVE_1H","ABOVE","ABOVE_1H_BIG","ABOVE_1H_STAMP","BUY_H_NEGATIVE","BUY_H_POSITIVE","EXTREME_ABOVE","INDICATOR_1","INDICATOR_2","INDICATOR_3","INDICATOR_4","INDICATOR_5","INDICATOR_6","INDICATOR_7","INDICATOR_8","INDICATOR_9","INDICATOR_10","INDICATOR_11","JUST_CROSS_ABOVE","MACD_DAY_CURVED","MACD_DAY_NEGATIVE","MACD_DAY_POSITIVE","MACD_DAY_POSITIVE_BY_2","MACD_DAY_POSITIVE_EXTREME","ROC_CURVED","ROC_CURVED_DAY_NEGATIVE","ROC_CURVED_DAY_POSITIVE","ROC_CURVED_HOUR_NEGATIVE","ROC_CURVED_HOUR_POSITIVE","ROC_CURVED_NOW","ROC_H","RSI_CURVED","RSI_CURVED_H","RSI_DOWN_H","RSI_H","RSI_UP","STILL_ABOVE","WHOLE_MARKET_CURVED"]

    for mickey in booleans_to_numeric:
        df[mickey] = df[mickey].apply(lambda x: 1 if x == True else 0)

    # single convert string "win" to 1 or "lost" to 0
    df["outcome"] = df["outcome"].apply(lambda x: 1 if x == "win" else 0)

    mapping_symbol = {
        'MKRUSDT': 10,
        'ASTRUSDT': 11,
        'BATUSDT': 12,
        'DYDXUSDT': 13,
        'EOSUSDT': 14,
        'ETCUSDT': 15,
        'LPTUSDT': 16,
        'MTLUSDT': 17,
        'QTUMUSDT': 18,
        'RLCUSDT': 19,
        'RPLUSDT': 20,
        'TRXUSDT': 21,
        'DARUSDT': 22,
        'IOSTUSDT': 23,
        'MANAUSDT': 24,
        'NEOUSDT': 25,
        'NKNUSDT': 26,
        'OCEANUSDT': 27,
        'ONTUSDT': 28,
        'STORJUSDT': 29,
        'XRPUSDT': 30,
        'BNXUSDT': 31,
        '1INCHUSDT': 32,
        'ONEUSDT': 33,
        'TLMUSDT': 34,
        'GMTUSDT': 35,
        'AGIXUSDT': 36,
        'ANKRUSDT': 37,
        'ATOMUSDT': 38,
        'DOGEUSDT': 39,
        'ENSUSDT': 40,
        'HIGHUSDT': 41,
        'HOTUSDT': 42,
        'IOTXUSDT': 43,
        'MATICUSDT': 44,
        'RENUSDT': 45,
        'SNXUSDT': 46,
        'THETAUSDT': 47,
        'TOMOUSDT': 48,
        'WAVESUSDT': 49,
        'AGLDUSDT': 50,
        'BTCUSDT': 51,
        'CROUSDT': 52,
        'LTCUSDT': 53,
        'ZECUSDT': 54,
        'ETHWUSDT': 55,
        'HFTUSDT': 56,
        'IMXUSDT': 57,
        'OPUSDT': 58,
        'SUNUSDT': 59,
        'ADAUSDT': 60,
        'ZRXUSDT': 61,
        'ANTUSDT': 62,
        'API3USDT': 63,
        'ARPAUSDT': 64,
        'ARUSDT': 65,
        'BANDUSDT': 66,
        'BCHUSDT': 67,
        'BOBAUSDT': 68,
        'CELOUSDT': 69,
        'CHZUSDT': 70,
        'WOOUSDT': 71,
        'CTSIUSDT': 72,
        'ICXUSDT': 73,
        'HNTUSDT': 74,
        'SXPUSDT': 75,
        'OMGUSDT': 76,
        'ALPHAUSDT': 77,
        'IDUSDT': 78,
        'ZILUSDT': 79,
        'ROSEUSDT': 80,
        'LQTYUSDT': 81,
        'ACHUSDT': 82,
        'JOEUSDT': 83,
        'HOOKUSDT': 84,
        'ARBUSDT': 85,
        'GFTUSDT': 86,
        'COTIUSDT': 87,
        'AKROUSDT': 88,
        'SSVUSDT': 89,
        'MAGICUSDT': 90,
        'BLURUSDT': 91,
        'FETUSDT': 92,
        'AUDIOUSDT': 93,
        'FXSUSDT': 94,
        'MASKUSDT': 95,
        'TRUUSDT': 96,
        'LOOKSUSDT': 97,
        'INJUSDT': 98,
        'RNDRUSDT': 99,
        'BELUSDT': 100,
        'LDOUSDT': 101,
        'DODOUSDT': 102,
        'CFXUSDT': 103,
        'GALAUSDT': 104,
        'STGUSDT': 105,
        'MINAUSDT': 106,
        'EGLDUSDT': 107,
        'CRVUSDT': 108,
        'SKLUSDT': 109,
        'GRTUSDT': 110,
        'C98USDT': 111,
        'BAKEUSDT': 112,
        'APTUSDT': 113,
        'LRCUSDT': 114,
        'LITUSDT': 115,
        'BLZUSDT': 116,
        'KSMUSDT': 117,
        'NEARUSDT': 118,
        'FLRUSDT': 119,
        'FTMUSDT': 120,
        'OGNUSDT': 121,
        'DENTUSDT': 122,
        'YGGUSDT': 123,
        'HBARUSDT': 124,
        'DUSKUSDT': 125,
        'GMXUSDT': 126,
        'FLMUSDT': 127,
        'ALICEUSDT': 128,
        'SANDUSDT': 129,
        'APEUSDT': 130,
        'ILVUSDT': 131,
        'VETUSDT': 132,
        'DOTUSDT': 133,
        'KNCUSDT': 134,
        'TRBUSDT': 135,
        'RUNEUSDT': 136,
        'AXSUSDT': 137,
        'DGBUSDT': 138,
        'ENJUSDT': 139,
        'SLPUSDT': 140,
        'COMPUSDT': 141,
        'KDAUSDT': 142,
        'AVAXUSDT': 143,
        'DASHUSDT': 144,
        'UNIUSDT': 145,
        'ZENUSDT': 146,
        'LINKUSDT': 147,
        'SFPUSDT': 148,
        'XEMUSDT': 149,
        'TWTUSDT': 150,
        'CVXUSDT': 151,
        'SUSHIUSDT': 152,
        'BALUSDT': 153,
        'XTZUSDT': 154,
        'TUSDT': 155,
        'CVCUSDT': 156,
        'AAVEUSDT': 157,
        'ALGOUSDT': 158,
        'CTKUSDT': 159,
        'YFIUSDT': 160,
        'KLAYUSDT': 161,
        'XLMUSDT': 162,
        'ETHUSDT': 163
        }

    # Map the categorical features to numerical values
    df['symbol.keyword'] = df['symbol.keyword'].replace(mapping_symbol)

    # Drop any rows that has NaN values in the dataset. We cannot have empty/NaN values
    df = df.dropna()

    # Normalize the "condition" string values to numerical 0 or 1 
    df = pd.get_dummies(data = df, columns = ["condition"], drop_first = True)

    # Let see how it looks
    print(df.to_string(index=False))
    print("****************")

    # Get 85% of the dataset
    train_df = df.sample(frac=0.85, random_state=417)

    # The rest of dataset is for testing (15%)
    test_df = df.drop(train_df.index)

    # Understand axis
    # axis=0: Refers to the index or rows. Operations are performed column-wise, meaning the operation is applied to each column.
    # axis=1: Refers to the columns. Operations are performed row-wise, meaning the operation is applied to each row.

    # The training dataset is all selected features (85%) except the target (what we'd like to predict)
    X_train = train_df.drop("outcome", axis=1)
    # The target is stored here
    y_train = train_df["outcome"]

    # Testing dataset (15%) except the target 
    X_test = test_df.drop("outcome", axis=1)
    # And target is stored here
    y_test = test_df["outcome"]

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearSVC(penalty="l2",loss="squared_hinge",C=10,random_state=417)

    # Train the datasets features vs target
    model.fit(X_train,y_train)

    # Preditct target ("outcome") with unused datasets 
    predictions = model.predict(X_test)

    # Calculate accuracy (optional)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {accuracy:.2%}")

    print("******** Predictions vs Actual ********")
    print(predictions)
    print(y_test.values)

    # See the features you are training on
    print("******** See the features you are training on ********")
    print(X_train.to_string(index=False))

    # Compare with the outcome you are training on
    print("******** Compare with the outcome you are training on ********")
    print(y_train.to_string(index=False))

    # See what you are testing with:
    print("******** See what you are testing with ********")
    print(X_test.to_string(index=False))

    # Compare with the outcome that should match/predict
    print("******** Compare with the outcome that should match/predict ********")
    print(y_test.to_string(index=False))

    #######  Using standard scaled #######

    # Train the datasets features using scaled datasets. Features vs outcome
    model.fit(X_train_scaled, y_train)

    # Get predictions using scaled test dataset. Predict outcome 
    y_pred = model.predict(X_test_scaled)

    # Calculate accuracy (optional)
    print("******** Prediction using scaled ********")
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2%}")

    print("******** Predictions vs Actual ********")
    print(y_pred)
    print(y_test.values)

    print("******** X_train_scaled datasets ********")
    print(pd.DataFrame(X_train_scaled).to_string(index=False))

    print("******** X_train datasets ********")
    print(pd.DataFrame(X_train).to_string(index=False))

    sleep(60)
