import os
import pandas as pd

AIR_QUALITY_PATH = "/data/air_quality/"


def load_data():
    df = pd.read_csv(
        os.path.join(AIR_QUALITY_PATH, 'AirQualityUCI.csv'),
        decimal=',', delimiter=';', parse_dates=[['Date', 'Time']],
        dayfirst=True)

    # pca = PCA(whiten=True)
    # pca.fit(df.iloc[:, 2:])

    return df


if __name__ == "__main__":
    data = load_data()
