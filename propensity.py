"""
Estimate the propensity score (i.e., CTR) with Logistic Matrix Factorization.
"""
import pickle
import os
import numpy as np
from FM import FM_Ehn
from preprocessor import Data


def type_confirm(data: Data) -> Data:
    return data


i_exp = 0
dataset = 'coat'

model_save_path = '../data/real-world/ctr/%s/%d/' % (dataset, i_exp)
data_save_path = '../data/real-world/%s/' % dataset
record_path = '../train_records/%s/ctr_train/' % (dataset)
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)
if not os.path.exists(data_save_path):
    os.makedirs(data_save_path)
if not os.path.exists(record_path):
    os.makedirs(record_path)

hypers = open(record_path + '%s_hyperparameter.txt' % (dataset), 'a+')


for i_exp in range(1):
    record = open(record_path + '%s_ctr_pred_exp_%d.txt' % (dataset, i_exp), 'a+')

    # Load the dataset
    path = "../data/real-world/%s/%s_cur0.data" % (dataset, dataset)
    file = open(path, "rb")
    obj = type_confirm(pickle.load(file))
    file.close()

    best_l2_reg_lambda = -1
    best_ce = float('inf')
    best_prediction = None
    best_prediction_val = None
    batch_size = 1024 #256#
    best_model = None

    args = {
        'learning_rate': 0.001,
        'embedding_size': 100,
        'num_user': obj.user_num,
        'num_item': obj.item_num,
        'bias_var_trade_off_weight': 0,
        'bias_weight': 0
    }
    l2_reg_lambda_list = [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]

    for i_lambda_ctr in range(1, 7):
        l2_reg_lambda = l2_reg_lambda_list[i_lambda_ctr]
        args['l2_reg_lambda'] = l2_reg_lambda
        mf = FM_Ehn(args)
        early_stop = 0
        local_best_ce = 10
        local_best_prediction = None
        local_best_prediction_val = None
        local_best_model = None
        epoch = 0
        while early_stop < 5:
            user, item, click, convert = obj.get_training_data(sample_ratio=4)
            train_num = user.shape[0]
            index = np.random.permutation(train_num)
            user, item, click = user[index], item[index], click[index]
            n_batch = train_num // batch_size + 1
            for batch in range(n_batch):
                user_id = user[batch*batch_size: min((batch+1)*batch_size, train_num)]
                item_id = item[batch*batch_size: min((batch+1)*batch_size, train_num)]
                y = click[batch*batch_size: min((batch+1)*batch_size, train_num)]

                _, _, _ = mf.train_batch(user_id, item_id, y)

            # Validation performance
            user, item, click, convert = obj.get_valid_data()
            user_id = user
            item_id = item
            y = click

            _, ctr_val, _, _, ce = mf.test(user_id, item_id, y)
            print("Cross entropy:", ce)
            record.write('Epoch = %d l2_reg_lambda = %f CE on the val set = %f\n' % (epoch, l2_reg_lambda, ce))

            # Predict the CTR
            user, item, click, convert = obj.get_training_data(sample_ratio=0)
            user_id = user
            item_id = item
            y = click

            _, ctr, _, _, _= mf.test(user_id, item_id, y)

            # Update early stopping
            if ce < local_best_ce:
                local_best_ce = ce
                local_best_prediction = ctr
                local_best_prediction_val = ctr_val
                early_stop = 0
                local_best_model = mf
            else:
                early_stop += 1
            epoch += 1

        if local_best_ce < best_ce:
            best_ce = local_best_ce
            best_l2_reg_lambda = l2_reg_lambda
            best_prediction = local_best_prediction
            best_prediction_val = local_best_prediction_val
            best_model = local_best_model
        print("Local predicted CTR:", local_best_prediction)
        record.write('\n')

    print("Best L2 reg lambda:", best_l2_reg_lambda)
    print("Predicted CTR:", best_prediction)
    obj.ctr_train = best_prediction

    file = open(data_save_path + "%s_cur%d.data" % (dataset, i_exp), 'wb')
    pickle.dump(obj, file)
    file.close()

    file = open(model_save_path + 'ctr_pred_%s_%d' % (dataset, i_exp), "wb")
    pickle.dump(best_prediction_val, file)
    pickle.dump(best_prediction, file)
    pickle.dump(best_model, file)
    file.close()

    record.write('\n\nBest l2_reg_lambda = %f\n' % (best_l2_reg_lambda))
    hypers.write('Model: %d Best l2_reg_lambda = %f\n' % (i_exp, best_l2_reg_lambda))
    record.close()

hypers.close()