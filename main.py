import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


if __name__ == "__main__":
    df = pd.read_csv("data/etmgeg_270.csv")
    print(df.corr())

    # overview of all data
    df.info()
    print(df.describe())

    # count and remove null-values
    print(df.isnull().sum())
    df.dropna()

    # save chosen columns to continue with
    df = df[["YYYYMMDD", "TG"]]
    df.info()
    print(df.describe())
    df.to_csv("data/useful_columns.csv", index=False)

    # continue with 2 chosen columns: YYYYMMDD = date (independent variable), TG = average day temperature (dependent variable)
    df2 = pd.read_csv("data/useful_columns.csv")
    print(df2.isnull().any())
    df2 = df2.dropna()
    print(df2.isnull().any())

    # correlation between date and temperature
    print(df.corr())

    # graph of date and temperature
    x_values = df2["YYYYMMDD"].values.reshape(-1, 1)
    y_values = df2["TG"]
    plt.scatter(x_values, y_values)
    plt.xlabel("Date")
    plt.ylabel("°C * 10^-1")
    plt.show()

    # splitting train and test data
    x_train, x_test, y_train, y_test = train_test_split(x_values, y_values, test_size=0.20, random_state=1)
    lr = LinearRegression()
    lr.fit(x_train, y_train)

    # visualization of train data
    plt.scatter(x_train, y_train)
    plt.plot(x_train, lr.predict(x_train), color="y")
    plt.xlabel("Date")
    plt.ylabel("°C * 10^-1")
    plt.show()

    # visualization of test data
    plt.scatter(x_test, y_test, color="r")
    plt.plot(x_test, lr.predict(x_test))
    plt.xlabel("Date")
    plt.ylabel("°C * 10^-1")
    plt.show()

    # calculate scores
    train_score = str(round(lr.score(x_train, y_train), 2))
    test_score = str(round(lr.score(x_test, y_test), 2))
    print(f"Score train data: {train_score}")
    print(f"Score test data: {test_score}")

    # prediction, ~11.1°C on 2023-06-21, 0.01% accurate
    print(lr.predict([[20230621]]))

    pickle.dump(lr, open("GUI/model_date_temperature.pkl", "wb"))
