import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
import category_encoders as ce
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split
import tensorflow as tf
import itertools
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from logit_regression import LogitRegression, LogitXGBRegression


DATA_FOLDER = os.path.join('..', 'data')


def get_transformed_column_names(df, cluster=False):
    transformers = [ce.BinaryEncoder(cols=[col]) for col in ['sex',
                                                            'includes_drug',
                                                            'facility_type',
                                                            'is_participant',
                                                            'professional_accepts_medicare_assignment',
                                                            'reported_quality_measures',
                                                            'committed_to_heart_health_through_the_million_hearts_initiative_',
                                                            'used_electronic_health_records'
                                                            ]]

    if not cluster:
        transformers.extend([ce.OneHotEncoder(cols=['department'])])

    for trans in transformers:
        df = trans.fit_transform(df)

    return df.columns

def replace_nans(df):

    df_numerical = df.select_dtypes(exclude=['object'])
    df_numerical.fillna(-1, inplace=True)

    df_categoric = df.select_dtypes(include=['object'])
    df_categoric.fillna('NONE', inplace=True)

    df = df_numerical.merge(df_categoric, left_index=True, right_index=True)

    return df

def get_train_test(df, test_size=0.33, seed=42, nnet=True):
    X = df.drop(['npi', 'avg_medicare_payment_amt', 'overcharge_ratio'], axis=1, inplace=False)
    if not nnet:
        X = X.drop(['hcpcs_code'], axis=1, inplace=False)
    Y = df['overcharge_ratio']
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=test_size, random_state=seed)

    return (X_Train, X_Test, Y_Train, Y_Test)


def get_encoders(model='other'):
    transformers = [(col, ce.BinaryEncoder(cols=[col])) for col in ['sex',
                                                                    'includes_drug',
                                                                    'facility_type',
                                                                    'is_participant',
                                                                    'professional_accepts_medicare_assignment',
                                                                    'reported_quality_measures',
                                                                    'committed_to_heart_health_through_the_million_hearts_initiative_',
                                                                    'used_electronic_health_records'
                                                                    ]]

    if model == 'cluster':
        return transformers
    elif model == 'tree':
        return (transformers + [('department', ce.OneHotEncoder(cols=['department'])),
                                ])
    else:
        return (transformers + [('department', ce.OneHotEncoder(cols=['department'])),

                                ('scaler', RobustScaler(with_centering=False))])

def get_betareg_model(df_claims, test_size, seed):

    df_claims = replace_nans(df_claims)
    X_Train, X_Test, Y_Train, Y_Test = get_train_test(df_claims, test_size, seed)

    encoders = get_encoders()

    # Using separate pipelines for transformer and estimator due to RFECV's bug #6321

    transformer_pipe = Pipeline(encoders )

    linear_model = RFECV(estimator=LinearRegression(), scoring = 'neg_mean_squared_error', step=1, cv=5)

    transformer_pipe.fit(X_Train)

    X_Train_transformed = transformer_pipe.transform(X_Train)
    X_Test_transformed = transformer_pipe.transform(X_Test)

    linear_model.fit(X_Train_transformed, Y_Train)

    linear_preds = linear_model.predict(X_Test_transformed)



    result = {'lreg_model': linear_model,
              'lreg_preds': linear_preds,
              'transformer': transformer_pipe,
              'features': get_transformed_column_names(X_Train)
              }

    return result

def get_lreg_model(df_claims, test_size, seed):

    df_claims = replace_nans(df_claims)
    X_Train, X_Test, Y_Train, Y_Test = get_train_test(df_claims, test_size, seed)

    encoders = get_encoders()

    # Using separate pipelines for transformer and estimator due to RFECV's bug #6321

    transformer_pipe = Pipeline(encoders )

    linear_model = RFECV(estimator=LinearRegression(), scoring = 'neg_mean_squared_error', step=1, cv=5)

    transformer_pipe.fit(X_Train)

    X_Train_transformed = transformer_pipe.transform(X_Train)
    X_Test_transformed = transformer_pipe.transform(X_Test)

    linear_model.fit(X_Train_transformed, Y_Train)

    linear_preds = linear_model.predict(X_Test_transformed)



    result = {'lreg_model': linear_model,
              'lreg_preds': linear_preds,
              'transformer': transformer_pipe,
              'features': get_transformed_column_names(X_Train)
              }

    return result

def get_breg_model(df_claims, test_size, seed):

    df_claims = replace_nans(df_claims)
    X_Train, X_Test, Y_Train, Y_Test = get_train_test(df_claims, test_size, seed)

    encoders = get_encoders()

    # Using separate pipelines for transformer and estimator due to RFECV's bug #6321

    transformer_pipe = Pipeline(encoders )

    linear_model = RFECV(estimator=LogitRegression(), scoring = 'neg_mean_squared_error', step=1, cv=5)

    transformer_pipe.fit(X_Train)

    X_Train_transformed = transformer_pipe.transform(X_Train)
    X_Test_transformed = transformer_pipe.transform(X_Test)

    linear_model.fit(X_Train_transformed, Y_Train)

    linear_preds = linear_model.predict(X_Test_transformed)



    result = {'lreg_model': linear_model,
              'lreg_preds': linear_preds,
              'transformer': transformer_pipe,
              'features': get_transformed_column_names(X_Train)
              }

    return result


def get_rforest_model(df_claims, test_size, seed):
    df_claims = replace_nans(df_claims)
    X_Train, X_Test, Y_Train, Y_Test = get_train_test(df_claims, test_size, seed, False)

    encoders = get_encoders('tree')

    forest_params = {'rforest__max_features': [ 0.7],
                     'rforest__min_samples_leaf': [200],
                     'rforest__n_estimators': [200],
                     'rforest__min_samples_split': [0.05],
                     'rforest__max_depth': [None]
                     }

    rforest_estimator = [('rforest', RandomForestRegressor(random_state=seed, oob_score=True))]

    rforest_pipe = Pipeline(encoders + rforest_estimator)

    rforest = GridSearchCV(estimator=rforest_pipe,
                           param_grid=forest_params,
                           scoring='neg_mean_squared_error',
                           iid=False,
                           refit=True,
                           cv=5)



    rforest.fit(X_Train, Y_Train)

    rforest_preds = rforest.predict(X_Test)

    result = {'rforest_model': rforest,
              'rforest_preds': rforest_preds,
              'features': get_transformed_column_names(X_Train)
              }

    return result


def get_xgboost_model(df_claims, test_size, seed):
    # df_claims = df_claims.drop(['committed_to_heart_health_through_the_million_hearts_initiative_', 'used_electronic_health_records'], axis=1)
    # replace_nans(df_claims, 'overcharge_ratio')

    df_claims = replace_nans(df_claims)
    X_Train, X_Test, Y_Train, Y_Test = get_train_test(df_claims, test_size, seed)

    encoders = get_encoders('tree')



    boost_params = {'xgboost__max_depth': [10],
                    'xgboost__gamma': [0.15],
                    'xgboost__subsample': [0.6],
                    'xgboost__colsample_bytree': [ 0.6],
                    'xgboost__learning_rate': [0.2]
                    }

    xgboost_estimator = [('xgboost', XGBRegressor(missing=-1))]
    xgboost_pipe = Pipeline(encoders + xgboost_estimator)

    xgboost_model = GridSearchCV(estimator=xgboost_pipe,
                                 param_grid=boost_params,
                                 scoring='neg_mean_squared_error',
                                 iid=False,
                                 refit=True,
                                 cv=5)



    xgboost_model.fit(X_Train, Y_Train)

    xgboost_preds = xgboost_model.predict(X_Test)

    result = {'xgboost_model': xgboost_model,
              'xgboost_preds': xgboost_preds,
              'features': get_transformed_column_names(X_Train)
              }

    return result

def nnet_cat_num_split(df, target_col):

    df_numerical = df.select_dtypes(exclude=['object'])
    df_numerical.fillna(-1, inplace=True)

    df_categoric = df.select_dtypes(include=['object'])
    df_categoric.fillna('NONE', inplace=True)

    df = df_numerical.merge(df_categoric, left_index=True, right_index=True)

    return df, df_numerical, df_categoric

def scale_and_preprocess(train_numerical, test_numerical, Y_Test, target_col):
    col_train_num = list(train_numerical.columns)
    col_train_num_bis = list(train_numerical.columns)



    col_train_num_bis.remove(target_col)

    mat_train = np.matrix(train_numerical)
    mat_test = np.matrix(test_numerical)
    mat_new = np.matrix(train_numerical.drop(target_col, axis=1))
    mat_y = np.array(train_numerical[target_col])

    mat_test_y = np.array(Y_Test)

    prepro_y = RobustScaler(with_centering=False)
    prepro_y.fit(mat_y.reshape(mat_y.shape[0], 1))

    transformed_y = pd.DataFrame(prepro_y.transform(mat_test_y.reshape(mat_test_y.shape[0], 1)), columns=[target_col])

    prepro = RobustScaler(with_centering=False)
    prepro.fit(mat_train)

    prepro_test = RobustScaler(with_centering=False)
    prepro_test.fit(mat_new)

    train_num_scale = pd.DataFrame(prepro.transform(mat_train), columns=col_train_num)
    test_num_scale = pd.DataFrame(prepro_test.transform(mat_test), columns=col_train_num_bis)

    return train_num_scale, test_num_scale, transformed_y, col_train_num, col_train_num_bis, prepro, prepro_y, prepro_test

def get_nnet_model(df_claims, test_size, seed, predict=False):

    LABEL = 'overcharge_ratio'

    X_Train, X_Test, Y_Train, Y_Test = get_train_test(df_claims, test_size, seed)

    train = pd.concat([X_Train, Y_Train], axis=1)
    train = train.reset_index(drop=True)

    test = pd.concat([X_Test], axis=1)
    test = test.reset_index(drop=True)

    train, train_numerical, train_categoric = nnet_cat_num_split(train, LABEL)
    train, test_numerical, test_categoric = nnet_cat_num_split(test, LABEL)

    col_train_cat = list(train_categoric.columns)

    train_num_scale, test_num_scale, test_y, col_train_num, col_train_num_bis, prepro, prepro_y, prepro_test = scale_and_preprocess(train_numerical, test_numerical, Y_Test, LABEL)

    train[col_train_num] = train_num_scale
    test[col_train_num_bis] = test_num_scale



    # List of features
    COLUMNS = col_train_num
    FEATURES = col_train_num_bis

    FEATURES_CAT = col_train_cat

    engineered_features = []

    for continuous_feature in FEATURES:
        engineered_features.append(
            tf.contrib.layers.real_valued_column(continuous_feature))

    for categorical_feature in FEATURES_CAT:
        sparse_column = tf.contrib.layers.sparse_column_with_hash_bucket(
            categorical_feature, hash_bucket_size=1000)

        engineered_features.append(
            tf.contrib.layers.embedding_column(sparse_id_column=sparse_column, dimension=16, combiner="sum"))

    # Training set and Prediction set with the features to predict
    training_set = train

    training_set[FEATURES_CAT] = training_set[FEATURES_CAT].applymap(str)
    test[FEATURES_CAT] = test[FEATURES_CAT].applymap(str)

    testing_set = pd.concat([test, test_y], axis=1)

    def input_fn_new(data_set, training=True):
        continuous_cols = {k: tf.constant(data_set[k].values) for k in FEATURES}

        categorical_cols = {k: tf.SparseTensor(
            indices=[[i, 0] for i in range(data_set[k].size)], values=data_set[k].values,
            dense_shape=[data_set[k].size, 1]) for k in FEATURES_CAT}

        # Merges the two dictionaries into one.
        feature_cols = dict(list(continuous_cols.items()) + list(categorical_cols.items()))

        if training == True:
            # Converts the label column into a constant Tensor.
            label = tf.constant(data_set[LABEL].values)

            # Returns the feature columns and the label.
            return feature_cols, label

        return feature_cols

    regressor = None

    ev = 0

    # Model

    regressor = tf.contrib.learn.DNNRegressor(feature_columns=engineered_features,
                                              dropout=0.2,
                                              activation_fn=tf.nn.relu, hidden_units=[1024, 1024, 1024],
                                              optimizer=tf.train.ProximalAdagradOptimizer(
                                                  learning_rate=0.15,
                                                  l1_regularization_strength=0.001
                                              ),
                                              model_dir=os.path.join(DATA_FOLDER, 'nnet_model_new_1'))
    if not predict:
        regressor.fit(input_fn=lambda: input_fn_new(training_set), steps=200)

    if predict:
        regressor.fit(input_fn=lambda: input_fn_new(training_set), steps=0)

    # categorical_cols = {
    # k: tf.SparseTensor(indices=[[i, 0] for i in range(training_set[k].size)], values=training_set[k].values,
    #                    dense_shape=[training_set[k].size, 1]) for k in FEATURES_CAT}



    # ev = regressor.evaluate(input_fn=lambda: input_fn_new(testing_set, training=True), steps=1)

    # loss_score4 = ev["loss"]
    # print("Final Loss on the testing set: {0:f}".format(loss_score4))

    y = regressor.predict(input_fn=lambda: input_fn_new(test, training=False))
    preds = list(itertools.islice(y, test.shape[0]))
    predictions = np.asarray(preds)
    print(predictions.shape)
    predictions = predictions.reshape(test.shape[0], 1)
    print(predictions.shape)
    y_predict_transformed = prepro_y.inverse_transform(predictions)

    # predictions = prepro_y.inverse_transform(predictions)
    #
    # y_predict = regressor.predict(input_fn=lambda: input_fn_new(test, training=False))
    # predictions = list(itertools.islice(y, test.shape[0]))
    # y_predict_transformed = prepro_y.inverse_transform(np.array(y_predict).reshape(test.shape[0], 1))

    result = {
              'nnet_preds': predictions,
              'nnet_test_y': Y_Test,
              'y_predict_transformed': y_predict_transformed
              }

    return result

def calculate_code_cluster_diams(df_claims, identifier_col, code_col):

    def compute_cutoff(level, cys):
        for i in range(len(cys), 0, -1):
            if cys[i - 1] < level:
                return i
        return -1

    def get_department_cutoff(df_cluster):
        counts, bins, ignored = plt.hist(df_cluster['code_cluster_diam'], bins=100)
        cumsums = np.cumsum(counts)
        max_cy = df_cluster.shape[0]
        strong_xcut = compute_cutoff(0.99 * max_cy, cumsums) / len(bins)
        mild_xcut = compute_cutoff(0.95 * max_cy, cumsums) / len(bins)
        return {'dept_strong_cut': strong_xcut,
                'dept_mild_cut': mild_xcut
                }

    EPSILON = 0.1

    df_physicians_and_cpts = pd.DataFrame(
        (df_claims.groupby(identifier_col)[code_col].unique().agg(lambda col: ' '.join(col))).reset_index())

    vec = CountVectorizer(min_df=1, binary=True)
    X = vec.fit_transform(df_physicians_and_cpts[code_col])
    sim = X.T * X
    df_physicians_and_cpts['code_cluster_diam'] = 0.0

    for row in range(0, X.shape[0]):
        codes = [code for code in X[row, :].nonzero()][1]
        dists = []
        for i, j in itertools.product(codes, codes):
            if i < j:
                sim_ij = sim.getrow(i).todense()[:, j][0]
                if sim_ij == 0:
                    sim_ij = EPSILON
                dists.append(1 / (sim_ij ** 2))

        df_physicians_and_cpts.at[row, 'code_cluster_diam'] = (0 if len(dists) == 0 else np.asscalar((np.sqrt(sum(dists)) / len(dists))))

    df_physician_department = pd.DataFrame(df_claims.groupby(code_col)['department'].unique().str[0]).reset_index()

    df_code_cluster = pd.merge(df_physicians_and_cpts, df_physician_department, on=identifier_col, how='inner')

    df_department_cutoff = df_code_cluster.groupby(df_code_cluster['department']).apply(get_department_cutoff).apply(
        pd.Series).reset_index()

    df_physicians_and_cpts = pd.merge(df_code_cluster, df_department_cutoff, on='department', how='inner')

    counts, bins, ignored = plt.hist(df_code_cluster['code_cluster_diam'], bins=100)
    cumsums = np.cumsum(counts)
    max_cy = df_code_cluster.shape[0]
    strong_xcut = compute_cutoff(0.99 * max_cy, cumsums) / len(bins)
    mild_xcut = compute_cutoff(0.95 * max_cy, cumsums) / len(bins)

    df_physicians_and_cpts['overall_strong_cut'] = strong_xcut
    df_physicians_and_cpts['overall_mild_cut'] = mild_xcut

    return (df_physicians_and_cpts)
