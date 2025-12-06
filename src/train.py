import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense
from tensorflow.keras.optimizers import Adam

def design_model(features):
    model = Sequential()
    num_features = features.shape[1]
    model.add(InputLayer(input_shape=(num_features, )))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    opt = Adam(learning_rate=0.01)
    model.compile(loss='mse', metrics=['mae'], optimizer=opt)
    return model

dataset = pd.read_csv('data/life_expectancy.csv')
dataset = dataset.drop(['Country'], axis=1)

labels = dataset.iloc[:, -1]
features = dataset.iloc[:, 0:20]
features = pd.get_dummies(features)

features_train, features_test, labels_train, labels_test = train_test_split(
    features, labels, test_size=0.33, random_state=23
)

numerical_features = features.select_dtypes(include=['float64', 'int64'])
numerical_columns = numerical_features.columns

ct = ColumnTransformer([('standardize', StandardScaler(), numerical_columns)], remainder='passthrough')

features_train_scaled = ct.fit_transform(features_train)
features_train_scaled = pd.DataFrame(features_train_scaled)

features_test_scaled = ct.transform(features_test)
features_test_scaled = pd.DataFrame(features_test_scaled)

my_model = design_model(features_train_scaled)
print(my_model.summary())

my_model.fit(features_train_scaled, labels_train, epochs=40, batch_size=1, verbose=1)

res_mse, res_mae = my_model.evaluate(features_test_scaled, labels_test, verbose=0)
print(res_mse, res_mae)
