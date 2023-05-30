"""
    MRDR-DL-GPL
"""
import os
import time
import pickle
import numpy as np
import pandas as pd
from FM import FM_Ehn, drFM
from pandas import DataFrame
from metric import dcg_at_k, ndcg_at_k, recall_at_k, type_confirm

i_exp= 0
dataset =  'coat' # 'yahoo' #

data_path = "../data/real-world/%s/%s_cur0.data" % (dataset, dataset)
file = open(data_path, "rb")
obj = type_confirm(pickle.load(file))
file.close()

record_file_path = '../train_records/mrdr/%s/jl/%d/' % (dataset, i_exp)
data_save_path = "../data/mrdr/%s/jl/%d/" % (dataset, i_exp)
out_statistics_path = '../excel/mrdr/%s/jl/%d/' % (dataset, i_exp)
if not os.path.exists(record_file_path):
    os.makedirs(record_file_path)
if not os.path.exists(data_save_path):
    os.makedirs(data_save_path)
if not os.path.exists(out_statistics_path):
    os.makedirs(out_statistics_path)

ctr_batch_size = 1024
dr_batch_size = 256
l2_reg_lambda_list = [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
user_num = obj.user_num
item_num = obj.item_num

args_ctr = {
    'learning_rate': 0.001,
    'embedding_size': 100,  # 64, 256, 128
    'num_user': user_num,
    'num_item': item_num,
    'bias_var_trade_off_weight': 7e-1,#1,#
    'bias_weight': 7e-1 #1e-1, #1, #
}

args = {
    'learning_rate': 0.001,
    'embedding_size': 100,  # 64, 256, 128
    'num_user': user_num,
    'num_item': item_num,
}

for i_lambda_ctr in range(1, 6):
    l2_reg_lambda_ctr = l2_reg_lambda_list[i_lambda_ctr]
    args_ctr['l2_reg_lambda'] = l2_reg_lambda_ctr
    for i_lambda_dr in range(3, 4):
        l2_reg_lambda_dr = l2_reg_lambda_list[i_lambda_dr]
        args['l2_reg_lambda'] = l2_reg_lambda_dr
        args['l2_reg_lambda_il'] = l2_reg_lambda_dr

        data_frm_all = {'l2_reg_lambda_ctr': [], 'l2_reg_lambda_cvr': [], 'CE': [],
                        'DCG@2': [], 'DCG@4': [], 'DCG@6': [],
                        'NDCG@2': [], 'NDCG@4': [], 'NDCG@6': [],
                        'Recall@2': [], 'Recall@4': [], 'Recall@6': []}

        record_path = record_file_path + 'cvr_pred_mrdr_jl_exp%d_%d_%d.txt' % (i_exp, i_lambda_ctr, i_lambda_dr)
        record = open(record_path, 'a+')
        for i in range(10):
            print('******************* Training Start... ******************* \n')
            print('******************* CTR Model Initializing... ******************* \n')

            record.write('********************************************************* \n')
            record.write('******************* Training Start... ******************* \n')
            record.write('********************************************************* \n\n')
            record.write('******************* CTR Model Initializing... ******************* \n')
            record.write('l2_reg_lambda_ctr: %f \n' % (l2_reg_lambda_ctr))

            ctr_mf = FM_Ehn(args_ctr) # Initialize the CTR prediction model
            ctr_early_stop = 0
            epoch = 0
            ctr_local_best_ce = 10
            ctr_local_best_pred = None
            ctr_best_model = None
            best_p_val = None
            while ctr_early_stop < 2:
                loss = 0
                start_time = time.time() * 1000.0

                user, item, click, convert = obj.get_training_data(sample_ratio=4)
                ctr_train_num = user.shape[0]
                index = np.random.permutation(ctr_train_num)
                user, item, click = user[index], item[index], click[index]
                n_batch_ctr = ctr_train_num // ctr_batch_size + ((ctr_train_num % ctr_batch_size) > 0)

                for batch in range(n_batch_ctr):
                    user_id = user[batch * ctr_batch_size: min((batch + 1) * ctr_batch_size, ctr_train_num)]
                    item_id = item[batch * ctr_batch_size: min((batch + 1) * ctr_batch_size, ctr_train_num)]
                    y = click[batch * ctr_batch_size: min((batch + 1) * ctr_batch_size, ctr_train_num)]
                    batch_loss, _, batch_bce = ctr_mf.train_batch(user_id, item_id, y)
                    loss += batch_loss
                    # print("Phase0: Initialize CTR  Batch: ", batch, " Loss: ", batch_loss)
                    record.write(
                        'Phase0: Get initial CTR Batch: %d  loss: %f  bce: %f\n' % (batch, batch_loss, batch_bce))
                end_time = time.time() * 1000.0

                loss /= n_batch_ctr
                print('Phase0: Initialize CTR l2_reg_lambda: ', l2_reg_lambda_ctr, "Exp ", i, " Epoch: ", epoch,
                      " Average Training loss: ", loss, "Training time: %d ms" % (end_time - start_time))
                record.write('Phase0: Initialize CTR model l2_reg_lambda: %f  Exp%d Epoch: %d  Average Training loss: %f Training time: %d ms\n' %
                             (l2_reg_lambda_ctr, i, epoch, loss, (end_time - start_time)))

                # Validation performance
                start_time = time.time() * 1000.0
                user_id, item_id, click, _ = obj.get_valid_data()
                _, p_val, _, ce, _ = ctr_mf.test(user_id, item_id, click)
                end_time = time.time() * 1000.0
                print("Validation time: %d ms" % (end_time - start_time))
                # print("l2_reg_lambda: ", l2_reg_lambda_ctr, " Cross entropy: ", ce)
                record.write('Phase0: Initialize CTR model l2_reg_lambda: %f  Validation Cross-entropy: %f Validation time: %d ms\n' %
                             (l2_reg_lambda_ctr, ce, (end_time - start_time)))

                # Update early stopping
                if ce < ctr_local_best_ce:
                    ctr_best_model = ctr_mf
                    ctr_local_best_ce = ce
                    # local_best_prediction = ctr
                    best_p_val = p_val
                    ctr_early_stop = 0
                else:
                    ctr_early_stop += 1

                record.write('Current best val ce for initial ctr model = %f Early stop = %d Epoch = %d \n' %
                             (ctr_local_best_ce, ctr_early_stop, epoch))
                epoch += 1
                # print("Local predicted CTR:", local_best_prediction)
                # torch.cuda.empty_cache()

            del (ctr_mf)
            ctr_mf = ctr_best_model
            # Predict the CTR
            start_time = time.time() * 1000.0
            user_id, item_id, click, _ = obj.get_training_data(sample_ratio=0)
            _, ctr_local_best_pred, _, _, _ = ctr_mf.test(user_id, item_id, click)
            end_time = time.time() * 1000.0
            obj.ctr_train = ctr_local_best_pred.detach().cpu().numpy()
            print("Test time: %d ms" % (end_time - start_time))
            # torch.cuda.empty_cache()
            print('******************* CTR Initialization Done. ******************* \n')
            record.write('******************* CTR Initialization Done. ******************* \n')


            print('******************* MRDR-DL Initialization ... ******************* \n')
            record.write('******************* MRDR-DL Initialization ... ******************* \n')
            dr_mf = drFM(args)
            dr_early_stop = 0
            epoch = 0
            dr_local_best_ce = 10
            dr_best_model = None
            best_delta_e = None
            val_best_delta_e = None
            while dr_early_stop < 2:
                loss = 0
                start_time = time.time() * 1000.0
                user, item, click, convert = obj.get_training_data(sample_ratio=0)
                train_num = user.shape[0]
                ctr = obj.ctr_train  # .detach().cpu().numpy()
                index = np.random.permutation(train_num)
                user, item, click, convert, ctr = user[index], item[index], click[index], convert[index], ctr[index]
                n_batch = train_num // dr_batch_size + 1
                dr_mf.embedding_copy() # Copy and train the imputation model
                for batch in range(n_batch):
                    user_id = user[batch * dr_batch_size: min((batch + 1) * dr_batch_size, train_num)]
                    item_id = item[batch * dr_batch_size: min((batch + 1) * dr_batch_size, train_num)]
                    y = convert[batch * dr_batch_size: min((batch + 1) * dr_batch_size, train_num)]
                    o = click[batch * dr_batch_size: min((batch + 1) * dr_batch_size, train_num)]
                    p = ctr[batch * dr_batch_size: min((batch + 1) * dr_batch_size, train_num)]
                    batch_loss = dr_mf.train_batch(user_id, item_id, y, o, p, 'Imputation Learning')
                    loss += batch_loss
                    record.write('Phase0: Initialize MRDR-DL model Epoch: %d Batch: %d  Imputation Learning Loss: %f\n' %
                                 (epoch, batch, batch_loss))
                end_time = time.time() * 1000.0
                loss /= n_batch
                print("Phase0: Initialize MRDR-DL Epoch: ", epoch, " Average Imputation Learning Loss: ", loss, "Training time: %d ms" % (end_time - start_time))
                record.write('Phase0: Initialize DR model Epoch: %d  Average Imputation Learning Loss: %f \n' % (epoch, loss))

                # Train the prediction model
                loss = 0
                start_time = time.time() * 1000.0
                user, item, click, convert = obj.get_training_data(sample_ratio=4)
                ctr = np.append(obj.ctr_train, np.array([1] * obj.missing_num))  # .detach().cpu().numpy()
                index = np.random.choice(user.shape[0], train_num, replace=False)
                user, item, click, convert, ctr = user[index], item[index], click[index], convert[index], ctr[index]
                n_batch = train_num // dr_batch_size + 1
                for batch in range(n_batch):
                    user_id = user[batch * dr_batch_size: min((batch + 1) * dr_batch_size, train_num)]
                    item_id = item[batch * dr_batch_size: min((batch + 1) * dr_batch_size, train_num)]
                    y = convert[batch * dr_batch_size: min((batch + 1) * dr_batch_size, train_num)]
                    o = click[batch * dr_batch_size: min((batch + 1) * dr_batch_size, train_num)]
                    p = ctr[batch * dr_batch_size: min((batch + 1) * dr_batch_size, train_num)]
                    batch_loss = dr_mf.train_batch(user_id, item_id, y, o, p, 'Prediction Learning')
                    loss += batch_loss
                    record.write('Phase0: Initialize MRDR-DL model Epoch: %d Batch: %d  Prediction Learning Loss: %f\n' %
                                 (epoch, batch, batch_loss))
                end_time = time.time() * 1000.0
                loss /= n_batch
                print("Phase0: Initialize MRDR-DL Epoch: ", epoch, " Average Prediction Learning Loss: ", loss, "Training time: %d ms" % (end_time - start_time))
                record.write('Phase0: Initialize DR model Epoch: %d  Average Prediction Learning Loss: %f \n' % (epoch, loss))

                # Validation performance
                user, item, click, convert = obj.get_valid_data()
                user_id, item_id, y, o, p = user, item, convert, click, [1] * user.shape[0]
                ce, _, val_delta_e = dr_mf.test(user_id, item_id, y, o, p)

                # Update early stopping
                if ce < dr_local_best_ce:
                    dr_early_stop = 0
                    dr_local_best_ce = ce
                    dr_best_model = dr_mf
                    val_best_delta_e = val_delta_e
                else:
                    dr_early_stop += 1

                record.write('Current local best ce for initial MRDR-DL model = %f Early stop = %d Epoch = %d\n' %
                             (dr_local_best_ce, dr_early_stop, epoch))
                epoch += 1

            del(dr_mf)
            dr_mf = dr_best_model
            # Test performance
            start_time = time.time() * 1000.0
            user_id, item_id, o, y = obj.get_training_data(sample_ratio=4)
            _, p, _, _, _ = ctr_mf.test(user_id, item_id, o)
            # p = [1] * user_id.shape[0]
            _, _, best_delta_e = dr_mf.test(user_id, item_id, y, o, p)
            end_time = time.time() * 1000.0
            print("Test time: %d ms" % (end_time - start_time))
            print('****************** MRDR-DL Initialization Done. ****************** \n')
            record.write('****************** MRDR-DL Initialization Done. ****************** \n')


            print('****************** Joint Learning Start ****************** \n')
            record.write('****************** Joint Learning Start ****************** \n')
            ctr_best_model = None
            dr_best_model = None
            ctr_best_pred = None
            cvr_best_pred = None
            epoch = 0
            jl_early_stop = 0
            dr_best_ce = float('inf')

            while jl_early_stop < 20:
                print('******************* CTR Learning ******************* \n')
                record.write('******************* CTR Learning ******************* \n')
                ctr_early_stop = 0
                ctr_local_best_ce = float('inf')
                ctr_local_best_model = None
                best_p_val = None
                delta_e_val = val_best_delta_e
                while ctr_early_stop < 3:
                    start_time = time.time() * 1000.0
                    user, item, click, _ = obj.get_training_data(sample_ratio=4)
                    delta_e_train = best_delta_e
                    delta_e_train[(obj.user_train.shape[0]):] = 0
                    ctr_train_num = user.shape[0]
                    index = np.random.permutation(ctr_train_num)
                    user, item, click, delta_e = user[index], item[index], click[index], delta_e_train[index]
                    n_batch_ctr = ctr_train_num // ctr_batch_size + ((ctr_train_num % ctr_batch_size) > 0)
                    loss = 0
                    for batch in range(n_batch_ctr):
                        user_id = user[batch * ctr_batch_size: min((batch + 1) * ctr_batch_size, ctr_train_num)]
                        item_id = item[batch * ctr_batch_size: min((batch + 1) * ctr_batch_size, ctr_train_num)]
                        y = click[batch * ctr_batch_size: min((batch + 1) * ctr_batch_size, ctr_train_num)]
                        delta_e_batch = delta_e[batch * ctr_batch_size: min((batch + 1) * ctr_batch_size, ctr_train_num)]
                        batch_loss, batch_bce_bias, batch_bce = ctr_mf.train_batch(user_id, item_id, y, delta_e_batch,
                                                                                   en_bias=True, en_var=True)
                        loss += batch_loss
                        # print("Phase0: Initialize CTR  Batch: ", batch, " Loss: ", batch_loss)
                        record.write(
                            'Exp: %d Phase 1 Joint Learning CTR, Epoch: %d Batch: %d  loss: %f  bce + bias: %f bce: %f\n' %
                            (i, epoch, batch, batch_loss, batch_bce_bias, batch_bce))
                    end_time = time.time() * 1000.0

                    loss /= n_batch_ctr
                    print('l2_reg_lambda: ', l2_reg_lambda_ctr, " Epoch: ", epoch, " Total Average Training loss: ", loss,
                          "Training time: %d ms" % (end_time - start_time))
                    record.write(
                        'Exp: %d l2_reg_lambda_ctr: %f  Epoch: %d  Average training loss: %f Training time: %d ms\n' % \
                        ( i, l2_reg_lambda_ctr, epoch, loss, (end_time - start_time)))

                    # Validation performance
                    start_time = time.time() * 1000.0
                    user_id, item_id, click, _ = obj.get_valid_data()
                    _, p_val, val_loss, ce, val_bce = ctr_mf.test(user_id, item_id, click, delta_e_val, True)
                    end_time = time.time() * 1000.0
                    print("l2_reg_lambda: ", l2_reg_lambda_ctr, " CTR Cross entropy (include bias): ", ce,
                          " CTR Cross entropy : ", val_bce,
                          "Validation time: %d ms" % (end_time - start_time))
                    record.write('Phase 1 Joint Learning CTR l2_reg_lambda: %f \n' % (l2_reg_lambda_ctr))
                    record.write('Phase 1 Joint Learning CTR Validation Loss: %f \n' % (val_loss))
                    record.write('Phase 1 Joint Learning CTR Current Validation CE(with bias) = %f \n' % (ce))
                    record.write('Phase 1 Joint Learning CTR Bias on the val set = %f \n' % (ce - val_bce))
                    record.write('Phase 1 Joint Learning CTR Current Validation CE = %f \n' % (val_bce))

                    if ce < ctr_local_best_ce:
                        ctr_local_best_model = ctr_mf
                        ctr_local_best_ce = ce
                        # obj.ctr_train = ctr_pred.detach().cpu().numpy()
                        best_p_val = p_val
                        ctr_early_stop = 0
                    else:
                        ctr_early_stop += 1
                    print('******************* CTR Learning Done. ******************* \n')
                    record.write('******************* CTR Learning Done. ******************* \n\n')

                del (ctr_mf)
                ctr_mf = ctr_local_best_model
                # Predict the CTR
                start_time = time.time() * 1000.0
                user_id, item_id, click, _ = obj.get_training_data(sample_ratio=0)
                _, ctr_pred, _, _, _ = ctr_mf.test(user_id, item_id, click)
                end_time = time.time() * 1000.0
                print("Local predicted CTR:", ctr_pred, "Test time: %d ms" % (end_time - start_time))
                obj.ctr_train = ctr_pred.detach().cpu().numpy()
                del (ctr_pred)

                print('******************* MRDR-DL Learning Start ******************* \n')
                record.write('****************** MRDR-DL Learning Start ****************** \n')
                dr_early_stop = 0
                dr_local_best_ce = float('inf') - 1
                dr_local_best_model = None
                while dr_early_stop < 3:
                    loss = 0
                    start_time = time.time() * 1000.0
                    user, item, click, convert = obj.get_training_data(sample_ratio=0)
                    train_num = user.shape[0]
                    ctr = obj.ctr_train  # .detach().cpu().numpy()
                    index = np.random.permutation(train_num)
                    user, item, click, convert, ctr = user[index], item[index], click[index], convert[index], ctr[index]
                    n_batch = train_num // dr_batch_size + 1
                    dr_mf.embedding_copy()  # Copy and train the imputation model
                    for batch in range(n_batch):
                        user_id = user[batch * dr_batch_size: min((batch + 1) * dr_batch_size, train_num)]
                        item_id = item[batch * dr_batch_size: min((batch + 1) * dr_batch_size, train_num)]
                        y = convert[batch * dr_batch_size: min((batch + 1) * dr_batch_size, train_num)]
                        o = click[batch * dr_batch_size: min((batch + 1) * dr_batch_size, train_num)]
                        p = ctr[batch * dr_batch_size: min((batch + 1) * dr_batch_size, train_num)]
                        batch_loss = dr_mf.train_batch(user_id, item_id, y, o, p, 'Imputation Learning')
                        loss += batch_loss
                        record.write('Phase 1 Joint Learning MRDR-DL Batch: %d  Imputation Learning Loss: %f\n' % (batch, batch_loss))

                    loss /= n_batch
                    end_time = time.time() * 1000.0
                    print("Phase 1 Joint Learning MRDR-DL Epoch: ", epoch, " Average Imputation Learning Loss: ", loss, "Training time: %d ms" % (end_time - start_time))
                    record.write('Phase 1 Joint Learning MRDR-DL l2_reg_lambda_dr: %f  Epoch: %d  Average Imputation Learning Loss: %f \n' %
                        (l2_reg_lambda_dr, epoch, loss))

                    # Train the prediction model
                    loss = 0
                    start_time = time.time() * 1000.0
                    user, item, click, convert = obj.get_training_data(sample_ratio=4)
                    ctr = np.append(obj.ctr_train, np.array([1] * obj.missing_num))  # .detach().cpu().numpy()
                    index = np.random.choice(user.shape[0], train_num, replace=False)
                    user, item, click, convert, ctr = user[index], item[index], click[index], convert[index], ctr[index]
                    n_batch = train_num // dr_batch_size + 1
                    for batch in range(n_batch):
                        user_id = user[batch * dr_batch_size: min((batch + 1) * dr_batch_size, train_num)]
                        item_id = item[batch * dr_batch_size: min((batch + 1) * dr_batch_size, train_num)]
                        y = convert[batch * dr_batch_size: min((batch + 1) * dr_batch_size, train_num)]
                        o = click[batch * dr_batch_size: min((batch + 1) * dr_batch_size, train_num)]
                        p = ctr[batch * dr_batch_size: min((batch + 1) * dr_batch_size, train_num)]
                        batch_loss = dr_mf.train_batch(user_id, item_id, y, o, p, 'Prediction Learning')
                        loss += batch_loss
                        record.write('Phase 1 Joint Learning MRDR-DL Batch: %d  Prediction Learning Loss: %f\n' % (batch, batch_loss))

                    end_time = time.time() * 1000.0
                    print("Phase 1 Joint Learning MRDR-DL Epoch: ", epoch, " Average Prediction Learning Loss: ", loss, "Training time: %d ms" % (end_time - start_time))
                    record.write('Phase 1 Joint Learning MRDR-DL l2_reg_lambda_dr: %f  Epoch: %d  Average Prediction Learning Loss: %f \n' %
                        (l2_reg_lambda_dr, epoch, loss))

                    # Validation performance
                    start_time = time.time() * 1000.0
                    user, item, click, convert = obj.get_valid_data()
                    user_id, item_id, y, o, p = user, item, convert, click, [1] * user.shape[0]
                    ce, _, val_delta_e = dr_mf.test(user_id, item_id, y, o, p)
                    end_time = time.time() * 1000.0
                    print("Exp: ", i, " Epoch: ", epoch, " Validation CE: ", ce, " Validation time: %d ms" % (end_time - start_time))
                    record.write('Phase 1 Joint Learning CVR l2_reg_lambda_cvr: %f  Epoch: %d Validation CE: %f \n' %
                        (l2_reg_lambda_dr, epoch, ce))

                    # Update early stopping
                    if ce < dr_local_best_ce:
                        dr_early_stop = 0
                        dr_local_best_ce = ce
                        dr_local_best_model = dr_mf
                        val_local_best_delta_e = val_delta_e
                    else:
                        dr_early_stop += 1

                del (dr_mf)
                dr_mf = dr_local_best_model
                # Test performance
                start_time = time.time() * 1000.0
                user_id, item_id, o, y = obj.get_training_data(sample_ratio=4)
                _, p, _, _, _ = ctr_mf.test(user_id, item_id, o)
                # p = [1] * user_id.shape[0]
                _, _, local_best_delta_e = dr_mf.test(user_id, item_id, y, o, p)
                end_time = time.time() * 1000.0
                print("Test time: %d ms" % (end_time - start_time))

                if dr_local_best_ce < dr_best_ce:
                    dr_best_ce = dr_local_best_ce
                    val_best_delta_e = val_local_best_delta_e
                    best_delta_e = local_best_delta_e

                    ctr_best_pred = obj.ctr_train
                    dr_best_model = dr_local_best_model
                    ctr_best_model = ctr_local_best_model

                    jl_early_stop = 0
                else :
                    jl_early_stop += 1

                record.write('****************** MRDR-DL Learning Done. ****************** \n')
                epoch += 1

            # CVR Prediction
            start_time = time.time() * 1000.0
            user_id, item_id, o, y = obj.get_test_data()
            _, p, _, _, _ = ctr_best_model.test(user_id, item_id, o)
            # p = [1] * user_id.shape[0]
            _, cvr_pred, _ = dr_best_model.test(user_id, item_id, y, o, p)
            end_time = time.time() * 1000.0
            print("Best model test time: %d ms" % (end_time - start_time))

            dcg = dcg_at_k(obj, cvr_pred.detach().cpu())
            ndcg = ndcg_at_k(obj, cvr_pred.detach().cpu())
            recall = recall_at_k(obj, cvr_pred.detach().cpu())
            print("Epoch:", epoch, "Cross entropy:", dr_best_ce, "DCG@2,4,6:", dcg, "NDCG@2,4,6:", ndcg,
                  "Recall@2,4,6:", recall)
            record.write(
                "\n\n Exp: %d l2_reg_lambda_ctr: %f l2_reg_lambda_cvr: %f Epoch: %d Cross entropy: %f DCG@2,4,6: %s NDCG@2,4,6: %s Recall@2,4,6: %s\n" %
                (i, l2_reg_lambda_ctr, l2_reg_lambda_dr, epoch, dr_best_ce, str(dcg), str(ndcg), str(recall)))


            file = open(data_save_path + "cvr_pred_%s_%d_%d_%d_%d" % (dataset, i_exp, i, i_lambda_ctr, i_lambda_dr),
                        "wb")  #
            pickle.dump(ctr_best_pred, file)
            pickle.dump(ctr_best_model, file)
            pickle.dump(dr_best_model, file)
            pickle.dump(dcg, file)
            pickle.dump(recall, file)
            pickle.dump(l2_reg_lambda_ctr, file)
            pickle.dump(l2_reg_lambda_dr, file)
            file.close()

            data_frm_all['l2_reg_lambda_ctr'].append(l2_reg_lambda_ctr)
            data_frm_all['l2_reg_lambda_cvr'].append(l2_reg_lambda_dr)
            data_frm_all['CE'].append(dr_best_ce.detach().cpu().numpy().item())
            data_frm_all['DCG@2'].append(dcg[0])
            data_frm_all['DCG@4'].append(dcg[1])
            data_frm_all['DCG@6'].append(dcg[2])
            data_frm_all['NDCG@2'].append(ndcg[0])
            data_frm_all['NDCG@4'].append(ndcg[1])
            data_frm_all['NDCG@6'].append(ndcg[2])
            data_frm_all['Recall@2'].append(recall[0])
            data_frm_all['Recall@4'].append(recall[1])
            data_frm_all['Recall@6'].append(recall[2])

            del (ctr_mf)
            del (dr_mf)
            del (ctr_best_model)
            del (dr_best_model)

        record.write('****************** End Training ****************** \n')
        record.close()

        data_all = DataFrame(data_frm_all)
        with pd.ExcelWriter(out_statistics_path + 'MRDR_DL_Jl_%s_%d_%d_%d.xlsx' %
                            (dataset, i_exp, i_lambda_ctr, i_lambda_dr)) as writer:
            data_all.to_excel(writer, sheet_name='data_total')

        del (data_frm_all)


