"""
    Baseline: MRDR; the real world dataset
"""
import os
import time
import pickle
import numpy as np
import pandas as pd
from FM import drFM
from pandas import DataFrame
from metric import dcg_at_k, ndcg_at_k, recall_at_k, type_confirm

i_exp_baseline_mrdr = 0
dataset = 'coat'
file = open("../data/real-world/%s/%s_cur0.data" % (dataset, dataset), 'rb')
obj = type_confirm(pickle.load(file))
file.close()
user_num = obj.user_num
item_num = obj.item_num
obj.ctr_train = obj.ctr_train.detach().cpu().numpy()

path = '../train_records/mrdr/%s/baseline/%d/' % (dataset, i_exp_baseline_mrdr)
data_save_path = '../data/mrdr/%s/baseline/%d/' % (dataset, i_exp_baseline_mrdr)
out_statistics_path = '../excel/mrdr/%s/baseline/%d/' % (dataset, i_exp_baseline_mrdr)
if not os.path.exists(path):
    os.makedirs(path)
if not os.path.exists(data_save_path):
    os.makedirs(data_save_path)
if not os.path.exists(out_statistics_path):
    os.makedirs(out_statistics_path)

batch_size = 256
l2_reg_lambda_list = [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
args = {
    'learning_rate': 0.001,
    'embedding_size': 100,
    'num_user': user_num,
    'num_item': item_num,
}

for i_lambda_cvr in range(1, 7):
    data_frm_all = {'neg_rate': [], 'l2_reg_lambda_ctr': [], 'l2_reg_lambda_cvr': [], 'l2_reg_lambda_cvr_il': [], 'CE': [],
                    'DCG@2': [], 'DCG@4': [], 'DCG@6': [], 'NDCG@2': [], 'NDCG@4': [], 'NDCG@6': [],
                    'Recall@2': [], 'Recall@4': [], 'Recall@6': []}
    args['l2_reg_lambda'] = l2_reg_lambda_list[i_lambda_cvr]
    args['l2_reg_lambda_il'] = l2_reg_lambda_list[i_lambda_cvr]
    record_best = open(path + 'cvr_mrdr_best_%d.txt' % i_lambda_cvr, 'a+')

    for i in range(10):
        file_path = path + 'cvr_mrdr_%s_%d_exp_%d.txt' % (dataset, i_lambda_cvr, i)
        record = open(file_path, 'a+')
        record.write('******************* %d-th exp: Training Start... ******************* \n' % i)
        mf = drFM(args)

        early_stop = 0
        epoch = 0
        best_ce = float('inf')
        best_dcg = None
        best_ndcg = None
        best_recall = None
        best_model = None
        best_prediction = None

        while early_stop < 10:
            loss = 0
            epoch += 1
            # Copy and train the imputation model
            mf.embedding_copy()
            start_time = time.time() * 1000.0
            user, item, click, convert = obj.get_training_data(sample_ratio=0)
            ctr_ = obj.ctr_train
            train_num = user.shape[0]
            n_batch = train_num // batch_size + 1
            index = np.random.permutation(train_num)
            user, item, click, convert, ctr_ = user[index], item[index], click[index], convert[index], ctr_[index]
            for batch in range(n_batch):
                user_id = user[batch * batch_size: min((batch + 1) * batch_size, train_num)]
                item_id = item[batch * batch_size: min((batch + 1) * batch_size, train_num)]
                y = convert[batch * batch_size: min((batch + 1) * batch_size, train_num)]
                o = click[batch * batch_size: min((batch + 1) * batch_size, train_num)]
                p = ctr_[batch * batch_size: min((batch + 1) * batch_size, train_num)]
                batch_loss = mf.train_batch(user_id, item_id, y, o, p, 'MRDR-IL')
                loss += batch_loss
                # print(" Batch: ", batch, " Loss: ", batch_loss)
                record.write('Batch: %d  MR Imputation Learning Loss: %f\n' % (batch, batch_loss))

            loss /= n_batch
            print("Epoch: ", epoch, " Average training loss: ", loss)
            record.write('Epoch: %d  Average training loss: %f \n' % (epoch, loss))

            # Train the prediction model
            loss = 0
            user, item, click, convert = obj.get_training_data(sample_ratio=4)
            ctr_ = np.append(obj.ctr_train, np.array([1] * obj.missing_num))  # .detach().cpu().numpy()
            index = np.random.choice(user.shape[0], train_num, replace=False)
            user, item, click, convert, ctr_ = user[index], item[index], click[index], convert[index], ctr_[index]
            n_batch = train_num // batch_size + 1
            for batch in range(n_batch):
                user_id = user[batch * batch_size: min((batch + 1) * batch_size, train_num)]
                item_id = item[batch * batch_size: min((batch + 1) * batch_size, train_num)]
                y = convert[batch * batch_size: min((batch + 1) * batch_size, train_num)]
                o = click[batch * batch_size: min((batch + 1) * batch_size, train_num)]
                p = ctr_[batch * batch_size: min((batch + 1) * batch_size, train_num)]
                batch_loss = mf.train_batch(user_id, item_id, y, o, p, 'Prediction Learning')
                loss += batch_loss
                # print(" Batch: ", batch, " Loss: ", batch_loss)
                record.write('Batch: %d  Prediction Learning Loss: %f\n' % (batch, batch_loss))

            end_time = time.time() * 1000.0
            print("Training time: ", (end_time - start_time))

            # Validation performance
            start_time = time.time() * 1000.0
            user, item, click, convert = obj.get_valid_data()
            user_id, item_id, y, o, p = user, item, convert, click, [1] * user.shape[0]
            ce, _, _ = mf.test(user_id, item_id, y, o, p)
            end_time = time.time() * 1000.0
            print("Validation time: ", (end_time - start_time))

            # Test performance
            start_time = time.time() * 1000.0
            user, item, click, convert = obj.get_test_data()
            user_id, item_id, y, o, p = user, item, convert, click, [1] * user.shape[0]
            _, prediction, _ = mf.test(user_id, item_id, y, o, p)
            end_time = time.time() * 1000.0
            print("Test time: ", (end_time - start_time))

            dcg = dcg_at_k(obj, prediction.detach().cpu())
            ndcg = ndcg_at_k(obj, prediction.detach().cpu())
            recall = recall_at_k(obj, prediction.detach().cpu())

            print("l2_reg_lambda_cvr: ",args['l2_reg_lambda'], "l2_reg_lambda_cvr_il: ",args['l2_reg_lambda_il'], "Exp: ", i,
                  " Epoch:", epoch, " Cross entropy:", ce.detach(),
                  " DCG@2,4,6:", dcg, " NDCG@2,4,6:", ndcg, " Recall@2,4,6:", recall)
            record.write("l2_reg_lambda_cvr: %f Exp: %d Epoch: %d Cross entropy: %f DCG@2,4,6: %s NDCG@2,4,6: %s Recall@2,4,6: %s\n" %
                         (args['l2_reg_lambda'], i, epoch, ce, str(dcg), str(ndcg), str(recall)))

            # Update early stopping
            if ce < best_ce:
                best_ce = ce.detach()
                best_dcg = dcg
                best_ndcg = ndcg
                best_recall = recall
                early_stop = 0
                best_prediction = prediction
                best_model = mf
                # best_i_lambda_cvr = i_lambda_cvr
            else:
                early_stop += 1
        print("l2_reg_lambda_cvr: ", args['l2_reg_lambda'], "l2_reg_lambda_cvr_il: ", args['l2_reg_lambda_il'], "Exp: ", i)
        print("Best cross entropy:", best_ce)
        print("Best DCG@2,4,6:", best_dcg)
        print("Best NDCG@2,4,6:", best_ndcg)
        print("Best Recall@2,4,6:", best_recall)

        record.write('****************** End Training ****************** \n')
        record.write("\n\n")
        record.write("Best cross entropy: %f Best DCG@2,4,6: %s Best DCG@2,4,6: %s Best Recall@2,4,6: %s\n\n" %
                     (best_ce, str(best_dcg), str(best_ndcg), str(best_recall)))
        record.close()

        # Predict the CTR
        file = open(data_save_path+ "cvr_pred_%s_data_%d_%d" % (dataset, i, i_lambda_cvr), "wb") #
        pickle.dump(best_prediction,file)
        pickle.dump(best_model, file)
        pickle.dump(best_dcg, file)
        pickle.dump(best_ndcg, file)
        pickle.dump(best_recall, file)
        file.close()

        del(mf)
        del(best_model)

        record_best.write("Best cross entropy: %f Best DCG@2,4,6: %s Best NDCG@2,4,6: %s Best Recall@2,4,6: %s (i_lambda_cvr = %d exp = %d)\n" % \
                          (best_ce, str(best_dcg), str(best_ndcg), str(best_recall), i_lambda_cvr, i))

        data_frm_all['neg_rate'].append(4)
        data_frm_all['l2_reg_lambda_ctr'].append(l2_reg_lambda_list[1])
        data_frm_all['l2_reg_lambda_cvr'].append(l2_reg_lambda_list[i_lambda_cvr])
        data_frm_all['l2_reg_lambda_cvr_il'].append(l2_reg_lambda_list[i_lambda_cvr])
        data_frm_all['CE'].append(best_ce.detach().cpu().numpy())
        data_frm_all['DCG@2'].append(best_dcg[0])
        data_frm_all['DCG@4'].append(best_dcg[1])
        data_frm_all['DCG@6'].append(best_dcg[2])
        data_frm_all['NDCG@2'].append(best_ndcg[0])
        data_frm_all['NDCG@4'].append(best_ndcg[1])
        data_frm_all['NDCG@6'].append(best_ndcg[2])
        data_frm_all['Recall@2'].append(best_recall[0])
        data_frm_all['Recall@4'].append(best_recall[1])
        data_frm_all['Recall@6'].append(best_recall[2])

    record_best.close()

    data_all = DataFrame(data_frm_all)
    with pd.ExcelWriter(out_statistics_path + 'baseline_mrdr_%s_%d_%d.xlsx' % (dataset, i_exp_baseline_mrdr, i_lambda_cvr)) as writer:
        data_all.to_excel(writer, sheet_name='data_total')

    del(data_frm_all)
    del(data_all)