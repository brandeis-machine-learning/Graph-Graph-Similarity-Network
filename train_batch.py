import numpy as np
from models import G2G_model, Discriminator
from datasets import Data_Loader
import os
import torch
import pandas as pd
import torch.nn.functional as F
import torch.optim as optim
import time
import math
import math
import argparse

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="MUTAG")
parser.add_argument('--lr', type=float, default=0.0003)
parser.add_argument('--epoch', type=int, default=500)
parser.add_argument('--threshold', type=float, default=0.5)
parser.add_argument('--alpha', type=float, default=1.0)
parser.add_argument('--beta', type=float, default=1.0)
parser.add_argument('--gamma', type=float, default=0.1)
args = parser.parse_args()


patience = args.epoch
DATASET = args.dataset
if DATASET == 'DD':
    LIMIT_NODE = 500
else:
    LIMIT_NODE = 0
BATCH_SIZE = 32
LEARNING_RATE = args.lr
DROP_RATE = 0.0
TRAIN_RATE = 0.9
SEED = 100
LOAD_MODEL = False
BATCH_SEPT = 1
SAVE_PER_EPOCHS = 50


np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

test_result, best_result, best_step, best_start, log_result = [], [], [], [], []
datasets = Data_Loader('data', DATASET, train_ratio=TRAIN_RATE, limit_nodes=LIMIT_NODE)
for k_fold in range(10):
    best_train, log_mean, step = 0.0, 0.0, 0
    log_result_k = []
    modle_save_file = 'history/' + DATASET + '_' + str(k_fold) + '_' + str(args.threshold) + '_' + str(args.alpha) + '_' + str(args.beta) + '_params.pkl'
    modle_disc_save_file = 'history/' + DATASET + '_' + str(k_fold) + '_' + str(args.threshold) + '_' + str(args.alpha) + '_' + str(args.beta) + '_params_disc.pkl'
    datasets.__load_data__(k_fold)
    label_name = datasets.label_name

    grm_model = G2G_model(datasets.nDims, len(label_name), dropout=DROP_RATE).cuda()
    discriminator = Discriminator(32, dropout=DROP_RATE).cuda()
    if LOAD_MODEL == True and os.path.exists(modle_save_file) and os.path.exists(modle_disc_save_file):
        grm_model.load_state_dict(torch.load(modle_save_file))
        discriminator.load_state_dict(torch.load(modle_disc_save_file))
    
    optimizer_G = optim.Adam(grm_model.parameters(), lr=LEARNING_RATE)#, weight_decay=0.001)
    optimizer_G_adv = optim.Adam(grm_model.parameters(), lr=LEARNING_RATE*args.gamma)#, weight_decay=0.001)
    optimizer_D = optim.SGD(discriminator.parameters(), lr=LEARNING_RATE*args.gamma)#, weight_decay=0.001)

    data_k_complete, adj_k_complete = [], []
    for i in range(len(datasets.data_train)):
        data_k_complete.append(datasets.data_train[i])
        adj_k_complete.append(datasets.adj_train[i])
    for i in range(len(datasets.data_test)):
        data_k_complete.append(datasets.data_test[i])
        adj_k_complete.append(datasets.adj_test[i])
    data_k_complete = np.array(data_k_complete)
    adj_k_complete = np.array(adj_k_complete)

    start_time = time.time()
    tol_step = 0
    while tol_step < patience:
        for batch_index in range(BATCH_SEPT):
            grmGenerator = datasets.grmGenerator(grm_model, batch_sept=BATCH_SEPT, batch_index=batch_index)
            batch_x, batch_a, batch_y, batch_real, batch_fake = next(grmGenerator)

            l1 = len(batch_x)
            l2 = len(datasets.data_test)
            y_train = np.argmax(batch_y, axis=1)
            y_test = np.argmax(np.array(datasets.label_test), axis=1)

            triplets_all = datasets.getTriplets(batch_y)
            indices = np.arange(len(triplets_all[0]))
            np.random.shuffle(indices)

            train_epochs = math.ceil(len(triplets_all[0]) / BATCH_SIZE)
            for epoch in range(train_epochs):
                source_idx = indices[epoch * BATCH_SIZE : (epoch + 1) * BATCH_SIZE]
                triplets = []
                for k_class in range(len(triplets_all)):
                    trip_k = []
                    for idx in source_idx:
                        trip_k.append(triplets_all[k_class][idx])
                    triplets.append(trip_k)
                # triplets = triplets_all
                for k_class in range(len(triplets)):
                    for index in range(len(triplets[k_class])):
                        a, p, n = triplets[k_class][index]
                        if index == 0:
                            disc_batch_real_a, disc_batch_real_p, disc_batch_real_n = batch_real[a:a+1], batch_real[p:p+1], batch_real[n:n+1]
                            disc_batch_fake_a, disc_batch_fake_p, disc_batch_fake_n = batch_fake[a:a+1], batch_fake[p:p+1], batch_fake[n:n+1]
                            gen_batch_x_a, gen_batch_x_p, gen_batch_x_n = batch_x[a:a+1], batch_x[p:p+1], batch_x[n:n+1]
                            gen_batch_a_a, gen_batch_a_p, gen_batch_a_n = batch_a[a:a+1], batch_a[p:p+1], batch_a[n:n+1]
                            gen_batch_y_a, gen_batch_y_p, gen_batch_y_n = batch_y[a:a+1], batch_y[p:p+1], batch_y[n:n+1]
                        else:
                            disc_batch_real_a = np.concatenate((disc_batch_real_a, batch_real[a:a+1]), axis=0).astype(float)
                            disc_batch_real_p = np.concatenate((disc_batch_real_p, batch_real[p:p+1]), axis=0).astype(float)
                            disc_batch_real_n = np.concatenate((disc_batch_real_n, batch_real[n:n+1]), axis=0).astype(float)
                            disc_batch_fake_a = np.concatenate((disc_batch_fake_a, batch_fake[a:a+1]), axis=0).astype(float)
                            disc_batch_fake_p = np.concatenate((disc_batch_fake_p, batch_fake[p:p+1]), axis=0).astype(float)
                            disc_batch_fake_n = np.concatenate((disc_batch_fake_n, batch_fake[n:n+1]), axis=0).astype(float)
                            gen_batch_x_a = np.concatenate((gen_batch_x_a, batch_x[a:a+1]), axis=0).astype(float)
                            gen_batch_x_p = np.concatenate((gen_batch_x_p, batch_x[p:p+1]), axis=0).astype(float)
                            gen_batch_x_n = np.concatenate((gen_batch_x_n, batch_x[n:n+1]), axis=0).astype(float)
                            gen_batch_a_a = np.concatenate((gen_batch_a_a, batch_a[a:a+1]), axis=0).astype(float)
                            gen_batch_a_p = np.concatenate((gen_batch_a_p, batch_a[p:p+1]), axis=0).astype(float)
                            gen_batch_a_n = np.concatenate((gen_batch_a_n, batch_a[n:n+1]), axis=0).astype(float)
                            gen_batch_y_a = np.concatenate((gen_batch_y_a, batch_y[a:a+1]), axis=0).astype(float)
                            gen_batch_y_p = np.concatenate((gen_batch_y_p, batch_y[p:p+1]), axis=0).astype(float)
                            gen_batch_y_n = np.concatenate((gen_batch_y_n, batch_y[n:n+1]), axis=0).astype(float)


                    disc_batch_real = np.concatenate((disc_batch_real_a, disc_batch_real_p, disc_batch_real_n), axis=0).astype(float)
                    disc_batch_fake = np.concatenate((disc_batch_fake_a, disc_batch_fake_p, disc_batch_fake_n), axis=0).astype(float)
                    gen_batch_x = np.concatenate((gen_batch_x_a, gen_batch_x_p, gen_batch_x_n), axis=0).astype(float)
                    gen_batch_a = np.concatenate((gen_batch_a_a, gen_batch_a_p, gen_batch_a_n), axis=0).astype(float)
                    gen_batch_y = np.concatenate((gen_batch_y_a, gen_batch_y_p, gen_batch_y_n), axis=0).astype(float)

                    real = torch.as_tensor(torch.from_numpy(np.ones([len(disc_batch_real), 1])), dtype=torch.float32).cuda()
                    fake = torch.as_tensor(torch.from_numpy(np.zeros([len(disc_batch_fake), 1])), dtype=torch.float32).cuda()
                    
                    # ============================================ generator loss
                    optimizer_G.zero_grad()
                    grm_model.train()
                    encode, z, decode, vec, pred, dist = grm_model(gen_batch_x, gen_batch_a)

                    # triplet_loss
                    pos_start = int(len(dist) / 3)
                    neg_start = pos_start * 2
                    similar_ap = torch.diag(dist[:pos_start, pos_start:neg_start])
                    similar_an = torch.diag(dist[:pos_start, neg_start:])
                    similar_pn = torch.diag(dist[pos_start:neg_start, neg_start:])
                    similar_aa = torch.diag(dist[:pos_start, :pos_start])
                    similar_pp = torch.diag(dist[pos_start:neg_start, pos_start:neg_start])
                    similar_nn = torch.diag(dist[neg_start:, neg_start:])

                    pos = similar_ap**2
                    neg = similar_an**2
                    minus = pos - neg + args.threshold
                    minus[minus < 0] = 0
                    trip_loss = torch.mean(minus)
                    dist_loss = torch.mean(((1-similar_ap)**2 + similar_an**2 + similar_pn**2) + (1-similar_aa)**2 + (1-similar_pp)**2 + (1-similar_nn)**2)

                    # mse_loss + class_loss
                    label = torch.as_tensor(np.argmax(gen_batch_y, axis=1), dtype=torch.long)
                    mse_loss = F.mse_loss(encode, decode)
                    class_loss = grm_model.loss(pred, label)

                    loss = mse_loss + args.alpha*class_loss + dist_loss + args.beta*trip_loss

                    loss.backward()
                    optimizer_G.step()

                    # ============================================ discriminator loss
                    optimizer_D.zero_grad()
                    discriminator.train()

                    
                    pred_from_real = discriminator(disc_batch_real)
                    pred_from_fake = discriminator(disc_batch_fake)
                    adv_loss_discriminator = -torch.mean(torch.log(pred_from_real) + torch.log(1 - pred_from_fake))
                    adv_loss_discriminator.backward()
                    optimizer_D.step()

                    discriminator.eval()
                    pred_from_real = discriminator(disc_batch_real)
                    pred_from_fake = discriminator(disc_batch_fake)


                    # ============================================ generator loss
                    grm_model.train()
                    optimizer_G_adv.zero_grad()
                    encode, z, decode, vec, pred, dist = grm_model(gen_batch_x, gen_batch_a)

                    grm_model.lockModel()
                    pred_from = discriminator(z)
                    adv_loss = -torch.mean(torch.log(1 - pred_from))
                    adv_loss.backward()
                    optimizer_G_adv.step()
                    grm_model.unlockModel()
                


            grm_model.eval()
            _, _, _, _, predict_train, _ = grm_model(batch_x,  batch_a)
            _, z, _, _, predict_test, _ = grm_model(datasets.data_test,  datasets.adj_test)

            predict_train = np.array(torch.argmax(predict_train, dim=1).detach().cpu())
            predict_test = np.array(torch.argmax(predict_test, dim=1).detach().cpu())

            count = 0.0
            for i in range(l1):
                if y_train[i] == predict_train[i]:
                    count += 1
            train_acc = count / l1
            count = 0.0
            for i in range(l2):
                if y_test[i] == predict_test[i]:
                    count += 1
            test_acc = count / l2
            
            log_result_k.append(test_acc)
            if tol_step > -1 and log_mean <= test_acc:
                if log_mean < test_acc:
                    start_step = tol_step
                best_train = train_acc
                log_mean = test_acc
                step = tol_step
            if (tol_step + 1) % SAVE_PER_EPOCHS == 0:

                if batch_index == BATCH_SEPT - 1:
                    _, _, _, vec, _, dist = grm_model(data_k_complete, adj_k_complete)
                    sim_save = dist.detach().cpu().numpy()
                    feat_save = vec.detach().cpu().numpy()
                    sim_file_50_save = 'history/' + DATASET + '_simi_' + str(k_fold) + '_' + str(tol_step + 1) + '_' + str(args.threshold) + '_' + str(args.alpha) + '_' + str(args.beta) + '_' + str(args.gamma) + '.csv'
                    feat_file_50_save = 'history/' + DATASET + '_feat_' + str(k_fold) + '_' + str(tol_step + 1) + '_' + str(args.threshold) + '_' + str(args.alpha) + '_' + str(args.beta) + '_' + str(args.gamma) + '.csv'
                    np.savetxt(sim_file_50_save, sim_save, delimiter = ',')
                    np.savetxt(feat_file_50_save, feat_save, delimiter = ',')

        if (tol_step + 1) % 10 == 0 or tol_step == 0:
            print('{:.0f} loss: {:.4f} mse_loss: {:.4f} dist_loss: {:.4f} trip_loss: {:.4f} time:{:.4f}'.format(tol_step+1, loss.data, mse_loss.data, dist_loss.data, trip_loss.data, time.time()-start_time))
        tol_step += 1
        
    mid_time = time.time()

    torch.save(grm_model.state_dict(), modle_save_file)
    torch.save(discriminator.state_dict(), modle_disc_save_file)
    print(k_fold, "finish training!!=====================================", mid_time - start_time)
    log_result.append(np.array(log_result_k))


log_result = np.array(log_result)
log_mean = np.mean(log_result, axis=0)


np.savetxt('history/' + DATASET + '_' + str(args.threshold) + '_' + str(args.alpha) + '_' + str(args.beta) + '_' + str(args.gamma) + '_result.csv', log_mean, delimiter = ',')
np.savetxt('history/' + DATASET + '_' + str(args.threshold) + '_' + str(args.alpha) + '_' + str(args.beta) + '_' + str(args.gamma) + '_result_std.csv', np.std(log_result, axis=0), delimiter = ',')