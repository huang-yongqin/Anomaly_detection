import os
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import random
import numpy.random as rnd
from sklearn.metrics import *


def save_results_csv(fname, results):
    new_rows = []
    if not os.path.isfile(fname):
        args = fname.split('/')[:-1]
        directory = os.path.join(*args)
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(fname, 'wt') as f:
            writer = csv.writer(f)
            writer.writerows(
                [['kpi', 'auc', 'accuracy', 'Precision', 'Recall', 'F1 score']])

    with open(fname, 'at') as f:
        # Overwrite the old file with the modified rows
        writer = csv.writer(f)
        new_rows.append(results)  # add the modified rows
        writer.writerows(new_rows)


# read data
def read_data(file_name):
    data = pd.read_csv(file_name, usecols=['value', 'label'])
    x_ = np.array(data['value']).reshape((len(data), 1))  # the dimesion of input_data is : 60000 rows, 1 column
    y_ = np.array(data['label'])
    return x_, y_


def handle_to_normal(data, label):
    indxs = np.where(label==0)[0]
    res = data[indxs]
    return res


def process_data(array_data, dim=5):
    processed_value = []
    for i in range(len(array_data)-dim + 1):
        processed_value.append(np.array(array_data[i:(i+dim)]))
    result = np.array(processed_value).reshape((len(processed_value), dim))
    return result


# statictical value for features
def handle_data(data, dim=5):
    res = []
    k = len(data)
    for i in range(k-dim+1):
        sub = data[i:(i+dim)]
        p = data[i+dim-1]
        t = [p, np.mean(sub), np.var(sub), abs(p-min(sub)), abs(p-max(sub))]
        res.append(t)
    res = np.array(res).reshape((k-dim+1, dim))
    return  res


# 丢弃历史窗口内具有异常标签的正常数据
def dropout_norm_with_anomaly(data, label, dim=5):
    k = len(data)
    data_res = []
    label_res = []
    for i in range(k - dim + 1):
        sub = np.array(data[i:(i + dim)])
        if label[i + dim - 1] == 1:
            label_res.append(1)
            data_res.append(sub)
        else:
            if sum(label[i:(i + dim)]) == 0:
                label_res.append(0)
                data_res.append(sub)
            else:
                continue
    new_len = len(label_res)
    return np.array(data_res).reshape((new_len, dim)), np.array(label_res).reshape((new_len,))



# 丢弃其历史窗口内异常数目达到一定比例的正常的数据
def process_test(data, label, dim=5):
    k = len(data)
    data_res = []
    label_res = []
    for i in range(k-dim+1):
        data_res.append(np.array(data[i:(i+dim)]))
        if label[i+dim-1]==1 and sum(label[i: i+dim-2, ])>=dim//3:
            label_res.append(1)
        elif label[i+dim-1]==0 and sum(label[i: i+dim-1])>=dim//2:
            label_res.append(1)
        else:
            label_res.append(0)
    return np.array(data_res).reshape((k-dim+1, dim)), np.array(label_res).reshape((k-dim+1, ))


def process(data, label, dim=5):
    k = len(data)
    data_res = []
    label_res = []
    m = dim//2
    for i in range(k - dim + 1):
        if label[i + dim - 1] == 1 :
            label_res.append(1)
            data_res.append(np.array(data[i:(i + dim)]))
        else:
            if sum(label[i:m])<= dim//4 and sum(label[m:(i + dim)])<=dim//8:
                label_res.append(0)
                data_res.append(np.array(data[i:(i + dim)]))
            else:
                continue
    new_len = len(label_res)
    return np.array(data_res).reshape((new_len, dim)), np.array(label_res).reshape((new_len,))


def get_threshold(array, percent=0.96):
    temp_array = array[:]
    temp_array.sort()
    n = int(len(temp_array)*percent)
    threshold = temp_array[n]
    return threshold


# def get_most_normal_instances(score, data):
#     threshold = get_threshold(score, percent=0.3).tolist()
#
#     df = pd.DataFrame(columns=['score', 'instance'])
#     df['score'] = np.array(score).reshape((len(score), ))
#     df['instance'] = data
#
#     res = df[df['score'] < threshold]['instance']
#     result = np.array(res).reshape((len(res), 1))
#     return result
#
#
# def query(instances):
#     feedback = []
#     for instance in instances:
#         df= pd.read_csv(file_name, usecols=['value', 'label'])
#         l = df['label'][df.value == instance[len(instance)-1]]
#         l = np.array(l)[0]
#         t = [l, instance]
#         feedback.append(t)
#     return feedback
#
#
# def get_query_instances(score, data, budget, k=2):
#     # score: numpy.ndarray
#     query_num = k
#     budget -= query_num
#     res = []
#     score = abs(score)
#
#     while query_num > 0:
#         query_num -= 1
#         v = min(score)
#         print(v)
#         indices = np.where(score==v)[0]
#         index = indices[0]  # tuple with[0] -> numpy.ndarray with[0][0] ->  numpy.int64
#
#         res.append(data[index])
#
#         score = np.delete(score, index)
#         data = np.delete(data, index, axis=0)
#     # print(np.array(res).shape)
#
#     return budget, np.array(res).reshape((len(res), data.shape[1]))


def get_label(prob, percentage=None):
    predict_label = []
    num_anom = 0

    if percentage is None:
        for p in prob:
            if p > 0.5:  # nominal
                predict_label.append(0)
            else:
                predict_label.append(1)
                num_anom += 1
    else:
        threshold = get_threshold(prob, percent=percentage)
        for p in prob:
            if p > threshold:  # anomaly
                predict_label.append(1)
                num_anom += 1
            else:
                predict_label.append(0)  # nominal

    return np.array(predict_label), num_anom


def measure(groundtruth, label):
    accuracy = accuracy_score(groundtruth, label)
    precision = precision_score(groundtruth, label)
    recall = recall_score(groundtruth, label)
    f_score = f1_score(groundtruth, label)

    return accuracy, precision, recall, f_score


def my_normalize(prob):
    max_v = max(prob)
    min_v = min(prob)
    d = max_v - min_v
    length = len(prob)
    # print('max-score - min-score', d)

    if d==0:
        result = np.zeros((length, 1))
    else:
        result = []
        for p in prob:
            normalize_p = (p - min_v) / d
            result.append(normalize_p)
        result = np.array(result)
    return result


def get_oppisite(prob):
    res = []
    for p in prob:
        res.append(-p)
    return res


def plot(x_value,  groundtruth, *label):
    all_ = [x_value, groundtruth, *label]
    all_data = []
    num_of_slice = 20
    step = int(len(x_value)/num_of_slice)

    for i in range(num_of_slice):
        part = []
        for j in range(3):
            if i == num_of_slice-1:
                p1 = all_[j][step*i: , ]
            else:
                p1 = all_[j][i*step: (i+1)*step, ]
            part.append(p1)
        all_data.append(part)

    for i in range(num_of_slice):
        if i==num_of_slice-1:
            x_label = range(step*i, len(x_value))
        else:
            x_label = range(step*i, step*(i+1))
        markers = ['+', '*']
        plot_label = ['groundtruth', 'result']

        plt.figure(figsize=(40, 16))
        for j in range(len(all_data[i])):
            if j==0:
                plt.subplot(211)
                plt.plot(x_label, all_data[i][j])
                plt.title('part_' + str(i))
            else:
                plt.subplot(212)
                plt.ylim(2, 6)
                plt.scatter(x_label, all_data[i][j]*(5-j), marker=markers[j-1], label=plot_label[j-1])
                plt.title('part_' + str(i) + '_result', fontsize=30)
        plt.savefig('figure/split/part_' + str(i) + '.png')
        plt.close()


# 想法：score=None，照常分割，然后单独画x和score两部分, 重新设置plot的布局，如果score为None，画布二分，否则三分
def plot_score(x, prob, score, y, label, file_name):
    name = file_name.split("/")[2].split(".")[0]
    file_path = 'figure/' + name
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    all_ = [x, prob, score, y, label]
    k = len(all_)

    all_data = []
    num_of_slice = 20
    step = int(len(x) / num_of_slice)

    for i in range(num_of_slice):
        part = []
        for j in range(k):
            if i == num_of_slice - 1:
                p1 = all_[j][step * i:, ]
            else:
                p1 = all_[j][i * step: (i + 1) * step, ]
            part.append(p1)
        all_data.append(part)

    for i in range(num_of_slice):
        if i == num_of_slice - 1:
            x_label = range(step * i, len(x))
        else:
            x_label = range(step * i, step * (i + 1))


        plt.figure(figsize=(40, 12))

        plt.subplot(411)
        plt.plot(x_label, all_data[i][0])
        plt.title('part_' + str(i))

        plt.subplot(412)
        plt.plot(x_label, all_data[i][1])
        plt.title('part_' + str(i) + '_prob', fontsize=30)

        plt.subplot(413)
        plt.plot(x_label, all_data[i][2])
        plt.title('part_' + str(i) + '_score', fontsize=30)

        markers = ['+', '*']
        colors = ['r', 'b']
        plot_label = ['groundtruth', 'result']
        plt.subplot(414)
        for j in range(3, len(all_data[i])):
            plt.scatter(x_label, all_data[i][j] * (8 - j),
                        marker=markers[j - 3], label=plot_label[j - 3], color=colors[j-3])
            plt.ylim(2, 8)
            plt.title('part_' + str(i) + '_result', fontsize=30)
        plt.savefig(file_path + '/part_' + str(i) + '.png')
        plt.close()


def plot_gen_data(real_data, gen_data, file_name):
    name = file_name.split("/")[2].split(".")[0]
    file_path = 'figure/' + name
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    x_ = range(len(real_data))
    plt.figure(figsize=(60, 16))
    plt.subplot(211)
    plt.plot(x_, real_data)
    plt.title('real data', fontsize=30)

    plt.subplot(212)
    plt.plot(gen_data)
    plt.title('gen data', fontsize=30)
    plt.savefig(file_path + '/real_gen.png')
    plt.close()


def plot_loss(loss_d, loss_g, file_name):
    name = file_name.split("/")[2].split(".")[0]
    file_path = 'figure/' + name
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    x_ = range(len(loss_d))
    plt.figure(figsize=(40, 16))
    plt.plot(x_, loss_d, label='loss_d', color='r')
    plt.plot(x_, loss_g, label='loss_g', color='b')
    plt.title('loss + gan + ' + name, fontsize=30)
    plt.xlabel('train num')
    plt.ylabel('loss value')
    plt.legend()
    plt.savefig(file_path + '/' + name + '_loss.png')
    plt.close()


def random_sample(data, num):
    length = len(data)
    indcies = random.sample(range(0, length), num)
    res = data[indcies]
    return res


def samples(model, session, data, sample_range, batch_size, num_points=10000, num_bins=100):

    # Return a tuple (db, pd, pg), where db is the current decision
    # boundary, pd is a histogram of samples from the data distribution,
    # and pg is a histogram of generated samples.

    xs = np.linspace(-sample_range, sample_range, num_points)
    bins = np.linspace(-sample_range, sample_range, num_bins)

    # decision boundary
    db = np.zeros((num_points, 1))
    for i in range(num_points // batch_size):
        db[batch_size * i:batch_size * (i + 1)] = session.run(
            model.D1,
            {
                model.x: np.reshape(
                    xs[batch_size * i:batch_size * (i + 1)],
                    (batch_size, 5)
                )
            }
        )

    # generated samples
    zs = np.linspace(-sample_range, sample_range, num_points)
    g = np.zeros((num_points, 1))
    for i in range(num_points // batch_size):
        g[batch_size * i:batch_size * (i + 1)] = session.run(
            model.G,
            {
                model.z: np.reshape(
                    zs[batch_size * i:batch_size * (i + 1)],
                    (batch_size, 1)
                )
            }
        )
    pg, _ = np.histogram(g, bins=bins, density=True)

    # data distribution
    if num_points > len(data):
        num_points = len(data)
    d = random_sample(data, num_points)
    pds, _ = np.histogram(d, bins=bins, density=True)

    return db, pds, pg


def plot_distributions(samps, sample_range, file_name):
    name = file_name.split("/")[2].split(".")[0]
    # name = 'kpi_12'
    file_path = 'figure/' + name
    # file_name = 'data/kpi/' + name + '.csv'
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    db, pds, pg = samps
    db_x = np.linspace(-sample_range, sample_range, len(db))
    p_x = np.linspace(-sample_range, sample_range, len(pds))
    f, ax = plt.subplots(1)
    ax.plot(db_x, db, label='decision boundary')

    plt.plot(p_x, pds, label='real data')
    plt.plot(p_x, pg, label='generated data')
    plt.title('1D Generative Adversarial Network + gan + ' + name, fontsize=30)
    plt.xlabel('Data values')
    plt.ylabel('Probability density')
    plt.legend()
    plt.savefig(file_path + '/' + name + '_data.png')
    plt.close()


def get_batches(x, batch_size, p):
    n = x.shape[0]
    indxs = range(n)
    batches = []
    for i in range(0, n, batch_size):
        et = min(i + batch_size, n)
        if i+batch_size < n:
            batches.append(x[indxs[i:et], :])
        else:
            batches.append(np.concatenate((x[indxs[i:et], :], np.zeros((p, x.shape[1])))))
    return batches


def get_train_batches(x, batch_size=-1, shuffle=False):
    n = x.shape[0]
    if batch_size < 0:
        batch_size = n
    indxs = np.arange(n)
    if shuffle:
        rnd.shuffle(indxs)
    for i in range(0, n, batch_size):
        et = min(i+batch_size, n)
        yield x[indxs[i:et], :]


if __name__ == '__main__':
    filename = 'data/kpi/kpi_1.csv'
    value, gtruth = read_data(filename)
    plot(value, gtruth, gtruth)