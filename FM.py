"""
The base model: Factorization Machine
"""
import torch
import numpy as np

class FM(torch.nn.Module):
    def __init__(self, arguments):
        super(FM, self).__init__()

        self.learning_rate = arguments['learning_rate']
        self.embedding_size = arguments['embedding_size']
        self.num_users = arguments['num_user']
        self.num_items = arguments['num_item']
        self.l2_reg_lambda = arguments['l2_reg_lambda']

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.all_users = torch.tensor(np.arange(self.num_users)).long().to(self.device)
        self.all_items = torch.tensor(np.arange(self.num_items)).long().to(self.device)

        self.user_factors = torch.nn.Embedding(self.num_users, self.embedding_size)
        self.user_factors.weight.data.normal_()
        self.item_factors = torch.nn.Embedding(self.num_items, self.embedding_size)
        self.item_factors.weight.data.normal_()

        self.bceLoss = torch.nn.BCEWithLogitsLoss()
        self.user_bias = torch.nn.Parameter(torch.randn(self.num_users), requires_grad=True)
        self.item_bias = torch.nn.Parameter(torch.randn(self.num_items), requires_grad=True)
        self.global_bias = torch.nn.Parameter(torch.randn(1), requires_grad=True)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        print('********************* MF Initialization Done *********************')

        self.to(self.device)

    def forward(self, user_id, item_id):
        # Get the dot product per row
        u = self.user_factors(user_id)
        v = self.item_factors(item_id)

        pred = (u * v).sum(axis=1)
        pred += self.user_bias[user_id] + self.item_bias[item_id] + self.global_bias

        ctr = torch.sigmoid(pred)

        return pred, ctr

    def totalLoss(self, pred, labels):
        bceLoss = self.bceLoss(pred, labels).mean()

        # print(self.user_factors(self.all_users))

        self.l2_regularization = torch.norm(self.user_factors(self.all_users)) + torch.norm(
            self.item_factors(self.all_items))
        self.l2_regularization += torch.norm(self.user_bias) + torch.norm(self.item_bias)
        self.l2_regularization += torch.norm(self.global_bias)

        loss = bceLoss + self.l2_reg_lambda * self.l2_regularization

        return loss, bceLoss

    def train_batch(self, user_input, item_input, label_input):
        # reset gradients
        self.optimizer.zero_grad()
        user_id = torch.Tensor(user_input).long().to(self.device)
        item_id = torch.Tensor(item_input).long().to(self.device)
        labels = torch.Tensor(label_input).float().to(self.device)  # click

        pred_prob, ctr_pred = self.forward(user_id, item_id)
        loss, bceLoss = self.totalLoss(pred_prob, labels)
        # backpropagate
        loss.backward()
        # update
        self.optimizer.step()

        return loss.item(), bceLoss.item()

    def test(self, user_input, item_input, label_input):
        self.eval()

        user_id = torch.Tensor(user_input).long().to(self.device)
        item_id = torch.Tensor(item_input).long().to(self.device)
        label_id = torch.Tensor(label_input).float().to(self.device)

        prediction, ctr = self.forward(user_id, item_id)
        testLoss, testBCELoss = self.totalLoss(prediction, label_id)

        return prediction, ctr, testLoss, testBCELoss

class FM_Ehn(torch.nn.Module):
    def __init__(self, arguments):
        super(FM_Ehn, self).__init__()

        self.learning_rate = arguments['learning_rate']
        self.embedding_size = arguments['embedding_size']
        self.num_users = arguments['num_user']
        self.num_items = arguments['num_item']
        self.l2_reg_lambda = arguments['l2_reg_lambda']
        self.alpha = arguments['bias_weight']
        self.beta = arguments['bias_var_trade_off_weight']

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.all_users = torch.tensor(np.arange(self.num_users)).long().to(self.device)
        self.all_items = torch.tensor(np.arange(self.num_items)).long().to(self.device)

        self.user_factors = torch.nn.Embedding(self.num_users, self.embedding_size)
        self.user_factors.weight.data.normal_(std=0.3)
        self.item_factors = torch.nn.Embedding(self.num_items, self.embedding_size)
        self.item_factors.weight.data.normal_(std=0.3)

        self.bceLoss = torch.nn.BCEWithLogitsLoss()
        self.user_bias = torch.nn.Parameter(torch.randn(self.num_users), requires_grad=True)
        self.item_bias = torch.nn.Parameter(torch.randn(self.num_items), requires_grad=True)
        self.global_bias = torch.nn.Parameter(torch.randn(1), requires_grad=True)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        print('********************* MF Initialization Done *********************')

        self.to(self.device)

    def forward(self, user_id, item_id):
        # Get the dot product per row
        u = self.user_factors(user_id)
        v = self.item_factors(item_id)

        pred = (u * v).sum(axis=1)
        pred += self.user_bias[user_id] + self.item_bias[item_id] + self.global_bias

        ctr = torch.sigmoid(pred)

        return pred, ctr

    def total_loss(self, pred_prob, ctr_pred, labels, cvr_errors = None, en_bias = False, en_var = False):
        loss = self.bceLoss(pred_prob, labels)
        bceLoss = loss.detach().item()
        if en_bias or en_var:
            e_hats = torch.Tensor(cvr_errors).float().to(self.device)
            ctr_temp = torch.clamp(ctr_pred, min = 1e-4)
            bias_var = (en_bias == True) * self.beta * ((labels - 2 * labels * ctr_temp + ctr_temp ** 2) * (torch.div(e_hats, ctr_temp) ** 2)).mean() + \
                       (en_var == True) * (1 - self.beta) * ((labels *(torch.div(e_hats, ctr_temp) ** 2)).mean()) / (labels.shape[0])
        else:
            # bias = 0
            bias_var = 0

        loss += self.alpha * bias_var
        bceLossWithBias = loss.detach().item()

        self.l2_regularization = torch.norm(self.user_factors(self.all_users)) + torch.norm(
            self.item_factors(self.all_items))
        self.l2_regularization += torch.norm(self.user_bias) + torch.norm(self.item_bias)
        self.l2_regularization += torch.norm(self.global_bias)

        loss += self.l2_reg_lambda * self.l2_regularization

        return loss, bceLossWithBias, bceLoss

    def train_batch(self, user_input, item_input, label_input, cvr_errors = None, en_bias = False, en_var = False):
        # self.train()
        # reset gradients
        self.optimizer.zero_grad()
        user_id = torch.Tensor(user_input).long().to(self.device)
        item_id = torch.Tensor(item_input).long().to(self.device)
        labels = torch.Tensor(label_input).float().to(self.device) # click

        pred_prob, ctr_pred = self.forward(user_id, item_id)
        loss, bce_bias, bceLoss = self.total_loss(pred_prob, ctr_pred, labels, cvr_errors, en_bias, en_var)
        # backpropagate
        loss.backward()
        # update
        self.optimizer.step()

        return loss.item(), bce_bias, bceLoss #bce_bias.item(), bceLoss.item()

    def test(self, user_input, item_input, label_input, cvr_errors = None, en_bias = False):
        user_id = torch.Tensor(user_input).long().to(self.device)
        item_id = torch.Tensor(item_input).long().to(self.device)
        label_id = torch.Tensor(label_input).float().to(self.device)

        with torch.no_grad():
            prediction, ctr_pred = self.forward(user_id, item_id)
            test_loss, test_bce_w_bias, test_bce = self.total_loss(prediction, ctr_pred, label_id, cvr_errors, en_bias)

        return prediction, ctr_pred, test_loss, test_bce_w_bias, test_bce

class weightFM(torch.nn.Module):
    def __init__(self, arguments):
        super(weightFM, self).__init__()
        self.learning_rate = arguments['learning_rate']
        self.embedding_size = arguments['embedding_size']
        self.num_users = arguments['num_user']
        self.num_items = arguments['num_item']
        self.l2_reg_lambda = arguments['l2_reg_lambda']

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.all_users = torch.tensor(np.arange(self.num_users)).long().to(self.device)
        self.all_items = torch.tensor(np.arange(self.num_items)).long().to(self.device)

        self.user_factors = torch.nn.Embedding(self.num_users, self.embedding_size)
        # self.user_factors.weight.data.uniform_(-0.05, 0.05)
        self.user_factors.weight.data.normal_(std=0.3)
        self.item_factors = torch.nn.Embedding(self.num_items, self.embedding_size)
        # self.item_factors.weight.data.uniform_(-0.05, 0.05)
        self.user_factors.weight.data.normal_(std=0.3)

        self.bceLoss = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.user_bias = torch.nn.Parameter(torch.randn(self.num_users), requires_grad=True)
        self.item_bias = torch.nn.Parameter(torch.randn(self.num_items), requires_grad=True)
        self.global_bias = torch.nn.Parameter(torch.randn(1), requires_grad=True)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        print('********************* MF Initialization Done *********************')

        self.to(self.device)

    def forward(self, user_id, item_id):
        # Get the dot product per row
        u = self.user_factors(user_id)
        v = self.item_factors(item_id)

        pred = (u * v).sum(axis=1)
        pred += self.user_bias[user_id] + self.item_bias[item_id] + self.global_bias
        cvr = torch.sigmoid(pred)

        return pred, cvr

    def totalLoss(self, y_hat, labels, ctr):
        bceLoss = self.bceLoss(y_hat, labels)
        ctr_temp = torch.clamp(ctr, min=1e-5)
        bceLoss = torch.div(bceLoss, ctr_temp)

        self.l2_regularization = torch.norm(self.user_factors(self.all_users)) + torch.norm(
            self.item_factors(self.all_items))
        self.l2_regularization += torch.norm(self.user_bias) + torch.norm(self.item_bias)
        self.l2_regularization += torch.norm(self.global_bias)

        loss = bceLoss.mean() + self.l2_reg_lambda * self.l2_regularization
        # loss = torch.div(bceLoss, ctr).mean() + self.l2_reg_lambda * self.l2_regularization

        return loss, bceLoss

    def train_batch(self, user_input, item_input, label_input, ctr_input):
        self.optimizer.zero_grad()
        user_id = torch.Tensor(user_input).long().to(self.device)
        item_id = torch.Tensor(item_input).long().to(self.device)
        label_id = torch.Tensor(label_input).float().to(self.device)
        ctr = torch.Tensor(ctr_input).float().to(self.device)

        pred, cvr_pred = self.forward(user_id, item_id)
        loss, _ = self.totalLoss(pred, label_id, ctr)
        # backpropagate
        loss.backward()
        # update
        self.optimizer.step()

        return loss.item()

    def test(self, user_input, item_input, label_input, ctr_input):
        user_id = torch.Tensor(user_input).long().to(self.device)
        item_id = torch.Tensor(item_input).long().to(self.device)
        label_id = torch.Tensor(label_input).float().to(self.device)
        ctr = torch.Tensor(ctr_input).float().to(self.device)
        with torch.no_grad():
            prediction, cvr_pred = self.forward(user_id, item_id)
            testLoss, testBCELoss = self.totalLoss(prediction, label_id, ctr)

        return prediction, cvr_pred, testLoss, testBCELoss

class drFM(torch.nn.Module):
    def __init__(self, arguments):
        super(drFM, self).__init__()

        self.learning_rate = arguments['learning_rate']
        self.embedding_size = arguments['embedding_size']
        self.num_users = arguments['num_user']
        self.num_items = arguments['num_item']
        self.l2_reg_lambda = arguments['l2_reg_lambda']
        self.l2_reg_lambda_il = arguments['l2_reg_lambda_il']

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.all_users = torch.tensor(np.arange(self.num_users)).long().to(self.device)
        self.all_items = torch.tensor(np.arange(self.num_items)).long().to(self.device)

        self.user_factors = torch.nn.Embedding(self.num_users, self.embedding_size)
        self.user_factors.weight.data.normal_(std=0.3)
        self.item_factors = torch.nn.Embedding(self.num_items, self.embedding_size)
        self.item_factors.weight.data.normal_(std=0.3)
        self.user_bias = torch.nn.Parameter(torch.randn(self.num_users), requires_grad=True)
        self.item_bias = torch.nn.Parameter(torch.randn(self.num_items), requires_grad=True)
        self.global_bias = torch.nn.Parameter(torch.randn(1), requires_grad=True)

        self.user_factors_il = torch.nn.Embedding(self.num_users, self.embedding_size)
        self.user_factors_il.weight.data.normal_(std = 0.3)
        self.item_factors_il = torch.nn.Embedding(self.num_items, self.embedding_size)
        self.item_factors_il.weight.data.normal_(std = 0.3)
        self.user_bias_il = torch.nn.Parameter(torch.randn(self.num_users), requires_grad=True)
        self.item_bias_il = torch.nn.Parameter(torch.randn(self.num_items), requires_grad=True)
        self.global_bias_il = torch.nn.Parameter(torch.randn(1), requires_grad=True)

        self.bceLoss = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        print('********************* MF Initialization Done *********************')

        self.to(self.device)

    def forward(self, user_id, item_id):
        # Get the dot product per row
        user_feature = self.user_factors(user_id)
        item_feature = self.item_factors(item_id)

        pred = (user_feature * item_feature).sum(axis=1)
        pred += self.user_bias[user_id] + self.item_bias[item_id] + self.global_bias

        cvr = torch.sigmoid(pred)

        return pred, cvr

    def forward_il(self, user_id, item_id):
        # Get the dot product per row
        user_feature_il = self.user_factors_il(user_id)
        item_feature_il = self.item_factors_il(item_id)

        pred_il = (user_feature_il * item_feature_il).sum(axis=1)
        pred_il += self.user_bias_il[user_id] + self.item_bias_il[item_id] + self.global_bias_il

        cvr_il = torch.sigmoid(pred_il)
        label = cvr_il.detach()

        return pred_il, cvr_il, label

    def imputationLearning(self, user_input, item_input, label_input, click_input, ctr_input):
        users = torch.Tensor(user_input).long().to(self.device)
        items = torch.Tensor(item_input).long().to(self.device)
        labels = torch.Tensor(label_input).float().to(self.device)
        clicks = torch.Tensor(click_input).float().to(self.device)
        ctr = torch.Tensor(ctr_input).float().to(self.device)

        pred, _, _ = self.forward_il(users, items)
        bce_loss = self.bceLoss(pred, labels)
        bce_loss = torch.div(bce_loss, ctr)
        bce_mrdr = torch.mul(bce_loss, torch.div(1-ctr, ctr))

        self.l2_regularization_il = torch.norm(self.user_factors_il(self.all_users)) + torch.norm(
            self.item_factors_il(self.all_items))
        self.l2_regularization_il += torch.norm(self.user_bias_il) + torch.norm(self.item_bias_il)
        self.l2_regularization_il += torch.norm(self.global_bias_il)

        loss_il = bce_loss.mean() + self.l2_reg_lambda_il * self.l2_regularization_il
        loss_il_mrdr = bce_mrdr.mean() + self.l2_reg_lambda_il * self.l2_regularization_il

        return loss_il, loss_il_mrdr

    def predLearning(self, user_input, item_input, label_input, click_input, ctr_input, mode = 'test'):
        users = torch.Tensor(user_input).long().to(self.device)
        items = torch.Tensor(item_input).long().to(self.device)
        labels = torch.Tensor(label_input).float().to(self.device)
        clicks = torch.Tensor(click_input).float().to(self.device)
        ctr = torch.Tensor(ctr_input).float().to(self.device)

        with torch.no_grad():
            _, _, predLabels = self.forward_il(users, items)

        pred, cvr = self.forward(users, items)
        error = self.bceLoss(pred, labels)
        error_il = self.bceLoss(pred, predLabels)

        if mode == 'train':
            dr_loss = torch.mul(error - error_il, torch.div(clicks, ctr)) + error_il

            self.l2_regularization = torch.norm(self.user_factors(self.all_users)) + torch.norm(
                self.item_factors(self.all_items))
            self.l2_regularization += torch.norm(self.user_bias) + torch.norm(self.item_bias)
            self.l2_regularization += torch.norm(self.global_bias)

            loss = dr_loss.mean() + self.l2_reg_lambda * self.l2_regularization
        else:
            loss = 0

        return loss, cvr, error.detach().mean(), (error - error_il).detach().cpu().numpy()

    def train_batch(self, user_input, item_input, label_input, click_input, ctr_input, mode = 'Imputation Learning'):
        # reset gradients
        self.optimizer.zero_grad()

        if mode == 'Prediction Learning':
            loss, _, _, _ = self.predLearning(user_input, item_input, label_input, click_input, ctr_input, 'train')
        elif mode == 'Imputation Learning':
            loss, _ = self.imputationLearning(user_input, item_input, label_input, click_input, ctr_input)
        elif mode == 'MRDR-IL':
            _, loss = self.imputationLearning(user_input, item_input, label_input, click_input, ctr_input)

        # backpropagate
        loss.backward()
        # update
        self.optimizer.step()

        return loss.item()

    def test(self, user_input, item_input, label_input,click_input, ctr_input):
        # user_id = torch.Tensor(user_input).long().to(self.device)
        # item_id = torch.Tensor(item_input).long().to(self.device)
        # label_id = torch.Tensor(label_input).float().to(self.device)
        with torch.no_grad():
            _, cvr, ce, delta_e = self.predLearning(user_input, item_input, label_input, click_input, ctr_input)

        return ce, cvr, delta_e

    def embedding_copy(self):

        self.user_factors_il.weight.data = self.user_factors.weight.detach().clone()
        self.item_factors_il.weight.data = self.item_factors.weight.detach().clone()
        self.user_bias_il.data = self.user_bias.detach().clone()
        self.item_bias_il.data = self.item_bias.detach().clone()
        self.global_bias_il.data = self.global_bias.detach().clone()


class drBiasFM(torch.nn.Module):
    def __init__(self, arguments):
        super(drBiasFM, self).__init__()

        self.learning_rate = arguments['learning_rate']
        self.embedding_size = arguments['embedding_size']
        self.num_users = arguments['num_user']
        self.num_items = arguments['num_item']
        self.l2_reg_lambda = arguments['l2_reg_lambda']
        self.l2_reg_lambda_il = arguments['l2_reg_lambda_il']
        self.dr_bias_lambda = arguments['lambda']

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.all_users = torch.tensor(np.arange(self.num_users)).long().to(self.device)
        self.all_items = torch.tensor(np.arange(self.num_items)).long().to(self.device)

        self.user_factors = torch.nn.Embedding(self.num_users, self.embedding_size)
        self.user_factors.weight.data.normal_(std=0.3)
        self.item_factors = torch.nn.Embedding(self.num_items, self.embedding_size)
        self.item_factors.weight.data.normal_(std=0.3)
        self.user_bias = torch.nn.Parameter(torch.randn(self.num_users), requires_grad=True)
        self.item_bias = torch.nn.Parameter(torch.randn(self.num_items), requires_grad=True)
        self.global_bias = torch.nn.Parameter(torch.randn(1), requires_grad=True)

        self.user_factors_il = torch.nn.Embedding(self.num_users, self.embedding_size)
        self.user_factors_il.weight.data.normal_(std = 0.3)
        self.item_factors_il = torch.nn.Embedding(self.num_items, self.embedding_size)
        self.item_factors_il.weight.data.normal_(std = 0.3)
        self.user_bias_il = torch.nn.Parameter(torch.randn(self.num_users), requires_grad=True)
        self.item_bias_il = torch.nn.Parameter(torch.randn(self.num_items), requires_grad=True)
        self.global_bias_il = torch.nn.Parameter(torch.randn(1), requires_grad=True)

        self.bceLoss = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        print('********************* MF Initialization Done *********************')

        self.to(self.device)

    def forward(self, user_id, item_id):
        # Get the dot product per row
        user_feature = self.user_factors(user_id)
        item_feature = self.item_factors(item_id)

        pred = (user_feature * item_feature).sum(axis=1)
        pred += self.user_bias[user_id] + self.item_bias[item_id] + self.global_bias

        cvr = torch.sigmoid(pred)

        return pred, cvr

    def forward_il(self, user_id, item_id):
        # Get the dot product per row
        user_feature_il = self.user_factors_il(user_id)
        item_feature_il = self.item_factors_il(item_id)

        pred_il = (user_feature_il * item_feature_il).sum(axis=1)
        pred_il += self.user_bias_il[user_id] + self.item_bias_il[item_id] + self.global_bias_il

        cvr_il = torch.sigmoid(pred_il)
        label = cvr_il.detach()

        return pred_il, cvr_il, label

    def imputationLearning(self, user_input, item_input, label_input, click_input, ctr_input):
        users = torch.Tensor(user_input).long().to(self.device)
        items = torch.Tensor(item_input).long().to(self.device)
        labels = torch.Tensor(label_input).float().to(self.device)
        clicks = torch.Tensor(click_input).float().to(self.device)
        ctr = torch.Tensor(ctr_input).float().to(self.device)

        pred, _, _ = self.forward_il(users, items)
        bce_loss = self.bceLoss(pred, labels)
        bce_loss = torch.div(bce_loss, ctr)
        bce_mrdr = torch.mul(bce_loss, torch.div(1-ctr, ctr))
        bce_dr_bias = bce_loss * clicks * (((clicks - ctr) / ctr) ** 2)

        self.l2_regularization_il = torch.norm(self.user_factors_il(self.all_users)) + torch.norm(
            self.item_factors_il(self.all_items))
        self.l2_regularization_il += torch.norm(self.user_bias_il) + torch.norm(self.item_bias_il)
        self.l2_regularization_il += torch.norm(self.global_bias_il)

        loss_il = bce_loss.mean() + self.l2_reg_lambda_il * self.l2_regularization_il
        loss_il_mrdr = bce_mrdr.mean() + self.l2_reg_lambda_il * self.l2_regularization_il

        loss_il_dr_mse =self.dr_bias_lambda * bce_dr_bias.mean() + (1 - self.dr_bias_lambda) * bce_mrdr.mean()  + \
                        self.l2_reg_lambda_il * self.l2_regularization_il

        return loss_il, loss_il_mrdr, loss_il_dr_mse

    def predLearning(self, user_input, item_input, label_input, click_input, ctr_input, mode = 'test'):
        users = torch.Tensor(user_input).long().to(self.device)
        items = torch.Tensor(item_input).long().to(self.device)
        labels = torch.Tensor(label_input).float().to(self.device)
        clicks = torch.Tensor(click_input).float().to(self.device)
        ctr = torch.Tensor(ctr_input).float().to(self.device)

        with torch.no_grad():
            _, _, predLabels = self.forward_il(users, items)

        pred, cvr = self.forward(users, items)
        error = self.bceLoss(pred, labels)
        error_il = self.bceLoss(pred, predLabels)

        if mode == 'train':
            dr_loss = torch.mul(error - error_il, torch.div(clicks, ctr)) + error_il

            self.l2_regularization = torch.norm(self.user_factors(self.all_users)) + torch.norm(
                self.item_factors(self.all_items))
            self.l2_regularization += torch.norm(self.user_bias) + torch.norm(self.item_bias)
            self.l2_regularization += torch.norm(self.global_bias)

            loss = dr_loss.mean() + self.l2_reg_lambda * self.l2_regularization
        else:
            loss = 0

        return loss, cvr, error.detach().mean(), (error - error_il).detach().cpu().numpy()

    def train_batch(self, user_input, item_input, label_input, click_input, ctr_input, mode = 'Imputation Learning'):
        # reset gradients
        self.optimizer.zero_grad()

        if mode == 'Prediction Learning':
            loss, _, _, _ = self.predLearning(user_input, item_input, label_input, click_input, ctr_input, 'train')
        elif mode == 'Imputation Learning':
            loss, _, _ = self.imputationLearning(user_input, item_input, label_input, click_input, ctr_input)
        elif mode == 'MRDR-IL':
            _, loss, _ = self.imputationLearning(user_input, item_input, label_input, click_input, ctr_input)
        elif mode == 'DR-MSE':
            _, _, loss = self.imputationLearning(user_input, item_input, label_input, click_input, ctr_input)
        # backpropagate
        loss.backward()
        # update
        self.optimizer.step()

        return loss.item()

    def test(self, user_input, item_input, label_input,click_input, ctr_input):
        # user_id = torch.Tensor(user_input).long().to(self.device)
        # item_id = torch.Tensor(item_input).long().to(self.device)
        # label_id = torch.Tensor(label_input).float().to(self.device)
        with torch.no_grad():
            # prediction, cvr = self.forward(user_id, item_id)
            # ce = self.bceLoss(prediction, label_id) # .mean()
            _, cvr, ce, delta_e = self.predLearning(user_input, item_input, label_input, click_input, ctr_input)

        return ce, cvr, delta_e

    def embedding_copy(self):

        self.user_factors_il.weight.data = self.user_factors.weight.detach().clone()
        self.item_factors_il.weight.data = self.item_factors.weight.detach().clone()
        self.user_bias_il.data = self.user_bias.detach().clone()
        self.item_bias_il.data = self.item_bias.detach().clone()
        self.global_bias_il.data = self.global_bias.detach().clone()