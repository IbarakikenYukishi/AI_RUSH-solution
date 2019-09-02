import argparse
import numpy as np
import os
import pandas as pd
import pickle
import xgboost as xgb
import nsml
from nsml import DATASET_PATH
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from pre3_final_fm import preprocess
from functools import partial
import optuna
import lightgbm as lgb

##########
num_models = 5  # 利用する学習機の数。
##########


def bind_nsml(model):
    def save(dir_name, *args, **kwargs):
        f_names = ['mymodel{}.pickle'.format(i) for i in range(num_models * 4)]
        for i in range(num_models * 4):
            os.makedirs(dir_name, exist_ok=True)
            f = open(os.path.join(dir_name, f_names[i]), 'wb')
            pickle.dump(model[i], f)
            f.close
        print('save model')

    def load(dir_name, *args, **kwargs):
        f_names = ['mymodel{}.pickle'.format(i) for i in range(num_models * 4)]
        for i in range(num_models * 4):
            f = open(os.path.join(dir_name, f_names[i]), 'rb')
            model[i] = pickle.load(f)
            f.close
        print('load model')

    def infer(root, phase):
        return _infer(root, phase, model=model)

    nsml.bind(save=save, load=load, infer=infer)


def _infer(root, phase, model):

    print('start infer')

    test_file = os.path.join(root, 'test', 'test_data', 'test_data')
    item = pd.read_csv(test_file,
                       dtype={'article_id': str,
                              'hh': int,
                              'gender': str,
                              'age_range': str,
                              'read_article_ids': str},
                       sep='\t')

    y_pred = np.zeros(len(item))
    fm = ['f', 'm']
    for i in range(2):
        for j in range(num_models * 2):
            myindex_test = (item['gender'] == fm[i]).values
            item_test = item.iloc[myindex_test, :]
            item_test = item_test.drop('gender', axis=1)
            item_test = preprocess(item=item_test, mode='test')
            dtest = xgb.DMatrix(item_test)

            ###################################
            # y_pred                          #
            # train と valid で書き換える !!!  #
            # train ... 上側  valid ... 下側  #
            ##################################

            # 学習の最終状態
            if j < num_models:  # xgbの場合
                y_pred[myindex_test] += model[j + i * num_models * 2].predict(dtest, ntree_limit=model[
                    j + i * num_models * 2].best_ntree_limit).ravel()  # early stopping の最良状態を提出する
            else:  # lgbの場合
                y_pred[myindex_test] += model[j + i * num_models * 2].predict(item_test, num_iteration=model[
                    j + i * num_models * 2].best_iteration).ravel()

    myindex_test_un = (item['gender'] == 'unknown').values
    if any(myindex_test_un):
        item_test = item.iloc[myindex_test_un, :]
        item_test = item_test.drop('gender', axis=1)
        item_test = preprocess(item=item_test, mode='test')
        dtest = xgb.DMatrix(item_test)

        for i in range(num_models * 4):
            if i < num_models:  # xgb
                y_pred[myindex_test_un] += 0.5 * \
                    model[i].predict(dtest, ntree_limit=model[
                                     i].best_ntree_limit).ravel()
            elif i < num_models * 2:  # lgb
                y_pred[myindex_test_un] += 0.5 * model[i].predict(
                    item_test, num_iteration=model[i].best_iteration).ravel()
            elif i < num_models * 3:  # xgb
                y_pred[myindex_test_un] += 0.5 * \
                    model[i].predict(dtest, ntree_limit=model[
                                     i].best_ntree_limit).ravel()
            else:  # lgb
                y_pred[myindex_test_un] += 0.5 * model[i].predict(
                    item_test, num_iteration=model[i].best_iteration).ravel()

    y_pred /= num_models * 2
    y_pred = y_pred.tolist()
    np.set_printoptions(threshold=np.inf)
    print(y_pred)
    print(len(y_pred))

    print('end infer')

    return y_pred


def evaluate(y_true, y_pred):
    """
    評価関数 (evaluation.py から転記してきたもの。)
    これを見るに、確率値で出力しても結局 0 or 1 に丸められてしまうと思われる。

    input: y_true, y_pred はともに numpy 配列.
    """
    y_pred[y_pred > 0.5] = 1
    y_pred[y_pred <= 0.5] = 0
    score = f1_score(y_true=y_true, y_pred=y_pred, pos_label=1)
    return score


# optuna用
def objective(dtrain, dvalid_x, y_true, evals, trial):
    # 調整してみたいパラメータ
    round_num = 500

    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'booster': 'gbtree',
        'lambda': trial.suggest_loguniform('lambda', 1e-8, 1.0),
        'alpha': trial.suggest_loguniform('alpha', 1e-8, 1.0)
    }

    if params['booster'] == 'gbtree' or params['booster'] == 'dart':
        params['tree_method'] = 'gpu_hist'
        params['max_depth'] = trial.suggest_int('max_depth', 1, 9)
        params['eta'] = trial.suggest_loguniform('eta', 1e-8, 1.0)
        params['gamma'] = trial.suggest_loguniform('gamma', 1e-8, 1.0)
        params['grow_policy'] = trial.suggest_categorical(
            'grow_policy', ['depthwise', 'lossguide'])
    if params['booster'] == 'dart':
        params['sample_type'] = trial.suggest_categorical(
            'sample_type', ['uniform', 'weighted'])
        params['normalize_type'] = trial.suggest_categorical(
            'normalize_type', ['tree', 'forest'])
        params['rate_drop'] = trial.suggest_loguniform('rate_drop', 1e-8, 1.0)
        params['skip_drop'] = trial.suggest_loguniform('skip_drop', 1e-8, 1.0)

    bst = xgb.train(params, dtrain, round_num,
                    early_stopping_rounds=10, evals=evals, verbose_eval=4)
    if params['booster'] != 'gblinear':
        y_pred = bst.predict(
            dvalid_x, ntree_limit=bst.best_ntree_limit).ravel()
    else:
        y_pred = bst.predict(dvalid_x).ravel()

    return 1.0 - evaluate(y_true, y_pred)


def main(args):
    """
    run, submit ともにこの関数 main が走るらしい。
    コマンドライン引数や nsml_bind の挙動が run と submit で変わることによって学習と予測が行われる感じ。
    """
    if args.mode == 'train':
        """
        訓練データ全体を使って学習
        (使ったことないからバグがあるかもしれない)
        """
        item_file = os.path.join(
            DATASET_PATH, 'train', 'train_data', 'train_data')
        label_file = os.path.join(DATASET_PATH, 'train', 'train_label')

        # item の取得 (共通)
        item = pd.read_csv(item_file,
                           dtype={'article_id': str,
                                  'hh': int,
                                  'gender': str,
                                  'age_range': str,
                                  'read_article_ids': str},
                           sep='\t')

        label = pd.read_csv(label_file, dtype={'label': int}, sep='\t')

        fm = ['f', 'm']
        model = [[], []]
        for i in range(2):
            myindex_train = (item['gender'] == fm[i]).values
            item_train = item.iloc[myindex_train, :]
            item_train = item_train.drop('gender', axis=1)
            item_train, label_train = preprocess(item=item_train, mode='train')
            dtrain = xgb.DMatrix(train_item, label=label_train)

            # xgboost の設計

            params = {}
            params['objective'] = 'binary:logistic'
            params['tree_method'] = 'gpu_hist'

            dtrain = xgb.DMatrix(item, label=label)

            num_round = 90
            model[i] = xgb.train(params, dtrain, num_round)

    elif args.mode == 'validation':
        """
        validation モード
        """

        item_file = os.path.join(
            DATASET_PATH, 'train', 'train_data', 'train_data')
        label_file = os.path.join(DATASET_PATH, 'train', 'train_label')

        # item の取得 (共通)
        item = pd.read_csv(item_file,
                           dtype={'article_id': str,
                                  'hh': int,
                                  'gender': str,
                                  'age_range': str,
                                  'read_article_ids': str},
                           sep='\t')

        label = pd.read_csv(label_file, dtype={'label': int}, sep='\t')

        item_train_0, item_valid_0, label_train_0, label_valid_0 = train_test_split(
            item, label)

        y_pred = np.zeros(len(item_valid_0))
        y_true = label_valid_0.values.ravel()

        fm = ['f', 'm']
        model = [0 for i in range(num_models * 4)]

        for i in range(2):#男女で場合分け
            for j in range(num_models * 2):#XGBとLGBMで場合分け
                print('##### gender: {} #####'.format(fm[i]))
                print('model:' + str(j + i * num_models * 2))
                if i == 0:  # 女性の場合
                    myindex_train = (item_train_0['gender'] == fm[i]).values
                    myindex_valid = (item_valid_0['gender'] == fm[i]).values
                    item_train = item_train_0.iloc[myindex_train, :]
                    label_train = label_train_0.iloc[myindex_train, :]
                    item_valid = item_valid_0.iloc[myindex_valid, :]
                    label_valid = label_valid_0.iloc[myindex_valid, :]
                    item_train = item_train.drop('gender', axis=1)
                    item_valid = item_valid.drop('gender', axis=1)
                    item_train, label_train = preprocess(
                        item=item_train, label=label_train, mode='train')
                    item_valid, label_valid = preprocess(
                        item=item_valid, label=label_valid, mode='validation')

                    item_valid_for_predict = item_valid
                if i == 1:  # 男性の場合
                    myindex_train = (item_train_0['gender'] == fm[i]).values
                    myindex_valid = (item_valid_0['gender'] == fm[i]).values
                    item_train = item_train_0.drop('gender', axis=1)
                    item_valid = item_valid_0.drop('gender', axis=1)
                    item_train, label_train = preprocess(
                        item=item_train, label=label_train_0, mode='train')
                    item_valid, label_valid = preprocess(
                        item=item_valid, label=label_valid_0, mode='validation')

                    item_valid_for_predict = item_valid_0.iloc[
                        myindex_valid, :]
                    label_valid_for_predict = label_valid_0.iloc[
                        myindex_valid, :]
                    item_valid_for_predict = item_valid_for_predict.drop(
                        'gender', axis=1)
                    item_valid_for_predict, label_valid_for_predict = preprocess(
                        item=item_valid_for_predict, label=label_valid_for_predict, mode='validation')

                xgb_dtrain = xgb.DMatrix(item_train, label=label_train)
                xgb_dvalid = xgb.DMatrix(item_valid, label=label_valid)
                evals = [(xgb_dtrain, 'train'), (xgb_dvalid, 'eval')]
                xgb_dvalid_x = xgb.DMatrix(item_valid)
                xgb_dvalid_for_predict = xgb.DMatrix(item_valid_for_predict)

                lgb_train = lgb.Dataset(item_train, label_train)
                lgb_valid = lgb.Dataset(
                    item_valid, label_valid, reference=lgb_train)
                lgb_dvalid_x = lgb.Dataset(item_valid)
                lgb_dvalid_for_predict = lgb.Dataset(item_valid_for_predict)

                num_round = 500

                if j < num_models:  # xgbの場合
                    ###paramsの設定###
                    ###すでにoptunaで最適化したものを仕様###
                    if i == 0:  # 女性の場合
                        xgb_params = {
                            'objective': 'binary:logistic',
                            'eval_metric': 'auc',
                            'booster': 'gbtree',
                            'lambda': 1.3201622516502017e-05,
                            'alpha': 1.7446882744649257e-05
                        }

                        xgb_params['tree_method'] = 'gpu_hist'
                        xgb_params['max_depth'] = 9
                        xgb_params['eta'] = 0.037114878438444764
                        xgb_params['gamma'] = 0.11912175821485932
                        xgb_params['grow_policy'] = 'depthwise'

                    else:  # 男性の場合
                        xgb_params = {
                            'objective': 'binary:logistic',
                            'eval_metric': 'auc',
                            'booster': 'gbtree',
                            'lambda': 3.1596746549472015e-06,
                            'alpha': 0.0028082312135669666
                        }

                        xgb_params['tree_method'] = 'gpu_hist'
                        xgb_params['max_depth'] = 7
                        xgb_params['eta'] = 0.036664410775719705
                        xgb_params['gamma'] = 0.1321100203613985
                        xgb_params['grow_policy'] = 'lossguide'

                    model[j + i * num_models * 2] = xgb.train(
                        xgb_params, xgb_dtrain, num_round, early_stopping_rounds=10, evals=evals, verbose_eval=4)
                    y_pred[myindex_valid] += model[j + i * num_models * 2].predict(
                        xgb_dvalid_for_predict, ntree_limit=model[j + i * num_models * 2].best_ntree_limit).ravel()

                else:  # lgbの場合
                    ###paramsの設定###
                    ###すでにoptunaで最適化したものを仕様###
                    if i == 0:  # 女性の場合
                        lgb_params = {
                            'objective': 'binary',
                            'metric': 'auc',
                            'boost_from_average': True,
                            'boosting_type': 'dart',
                            'num_leaves': 116,
                            'learning_rate': 0.08369183585733245
                        }

                        lgb_params['drop_rate'] = 1.1512686324655307e-07
                        lgb_params['skip_drop'] = 0.022683002949155356

                    else:  # 男性の場合
                        lgb_params = {
                            'objective': 'binary',
                            'metric': 'auc',
                            'boost_from_average': True,
                            'boosting_type': 'goss',
                            'num_leaves': 834,
                            'learning_rate': 0.02950367077219803
                        }

                        lgb_params['top_rate'] = 0.7414948856393442
                        lgb_params['other_rate'] = 0.20299571668477095

                    model[j + i * num_models * 2] = lgb.train(lgb_params, lgb_train, num_round,
                                                              early_stopping_rounds=10, valid_sets=lgb_valid, verbose_eval=4)

                    y_pred[myindex_valid] += model[j + i * num_models * 2].predict(item_valid_for_predict, num_iteration=model[
                        j + i * num_models * 2].best_iteration).ravel()

        # validation に対する F値
        print('validation に対する F値')
        print(evaluate(y_true, y_pred))

    else:  # テストモード
        model = [0 for i in range(num_models * 4)]

    bind_nsml(model)

    if args.pause:
        nsml.paused(scope=locals())

    nsml.save('uesakasumire_wa_kawaii')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # train または validation を選ぶ
    parser.add_argument("--mode", type=str, default='validation')
    parser.add_argument("--pause", type=int, default=0)

    config = parser.parse_args()

    main(config)
