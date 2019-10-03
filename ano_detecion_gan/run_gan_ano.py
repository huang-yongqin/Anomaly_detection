import warnings
warnings.filterwarnings('ignore')

import argparse
import tensorflow as tf
from rstl import STL
from gan import GAN
from gan_util import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-steps', type=int, default=10000,
                        help='the number of training steps to take')
    parser.add_argument('--hidden-size', type=int, default=4,
                        help='MLP hidden size')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='the batch size')
    parser.add_argument('--minibatch', action='store_true',
                        help='use minibatch discrimination')
    parser.add_argument('--log-every', type=int, default=10,
                        help='print loss after this many steps')
    parser.add_argument('--anim-path', type=str, default=None,
                        help='path to the output animation file')
    parser.add_argument('--anim-every', type=int, default=1,
                        help='save every Nth frame for animation')
    parser.add_argument('--dim', type=int, default=1,
                        help='the dimession of an instance')
    parser.add_argument('--file_name', type=str,
                        help='file name for data-set')
    return parser.parse_args()


def run(file_name, iter):
    args = parse_args()
    args.file_name = file_name
    x, y = read_data(file_name)  # 1D

    data = pd.read_csv(file_name, usecols=['timestamp', 'value'])
    decomp = STL(data.value, freq=1440, s_window="periodic", robust=True)
    x = decomp.remainder

    length = int(len(x) * 0.7)

    train_x = x[:length, ]
    train_y = y[:length, ]
    test_x = x[length:, ]
    test_y = y[length:, ]

    copy_train_x = train_x  # 1D with anomaly

    w = 20

    # # first 1D to *D, then choose point with label=0 as training set
    # train_x_fit = process_data(copy_train_x, dim=w)
    # train_y_fit = train_y[w-1:, ]
    # train_x = handle_to_normal(train_x_fit, train_y_fit)
    # test_x = process_data(test_x, dim=w)
    # test_y = test_y[w-1:, ]

    # for this way, dropout some points with many anomaly in their history windows
    train_x_fit, train_y_fit = process(copy_train_x, train_y, dim=w)
    train_x = handle_to_normal(train_x_fit, train_y_fit)  # choose points with label=0 as training set
    test_x, test_y = process(test_x, test_y, dim=w)

    args.dim = train_x.shape[1]
    print(args.dim)
    gan = GAN(args)
    gan.train(train_x)


    """ for test """
    prob, _ = gan.get_prob(test_x)  # output of D1 and D2
    label, num_ano_pred = get_label(prob)
    normalize_prob = my_normalize(get_oppisite(prob))  # normalize prob to make anomaly a high score

    # plot for test data
    # plot_score(test_x[:, args.dim - 1], prob, normalize_prob, test_y, label, file_name)
    num_anom_true = 0
    for l in test_y:
        if l == 1:
            num_anom_true += 1
    print()
    print('test 内真实的异常数目', num_anom_true)
    print('test 内检测到的异常数目: ', num_ano_pred)
    print()
    fpr, tpr, thresholds = roc_curve(test_y, normalize_prob)
    auc_s = auc(fpr, tpr)
    acc, precision, recall, f1 = measure(test_y, label)
    print('auc={}  accuracy={}   precision={}   recall={}   f1-score={}'.format(auc_s, acc, precision, recall, f1))
    res = [file_name, auc_s, acc, precision, recall, f1]
    save_results_csv('result/test_dim_20_label.csv', res)


    """  for train """
    print('---------train -----------')
    train_prob, _ = gan.get_prob(train_x_fit)  # output of D1 and D2
    train_label, train_num_ano_pred = get_label(train_prob)
    train_normalize_prob = my_normalize(get_oppisite(train_prob))  # normalize prob to make anomaly a high score

    train_num_anom_true = 0
    for l in train_y:
        if l == 1:
            train_num_anom_true += 1

    print()
    print('train 内真实的异常数目', train_num_anom_true)
    print('train 内检测到的异常数目: ', train_num_ano_pred)
    print()
    fpr_, tpr_, _ = roc_curve(train_y_fit, train_normalize_prob)
    train_auc_s = auc(fpr_, tpr_)
    acc_, precision_, recall_, f1_ = measure(train_y_fit, train_label)
    print('auc={}  accuracy={}   precision={}   recall={}   f1-score={}'.format(train_auc_s, acc_, precision_, recall_, f1_))
    res = [file_name, train_auc_s, acc_, precision_, recall_, f1_]
    save_results_csv('result/train_dim_20_label.csv', res)

    # for generated data
    # k = 20
    # real_data = random_sample(train_x, k)
    # gen_data = gan.get_gen_out(k)[0]
    # for i in range(k):
    #     plot_gen_data(real_data[i], gen_data[i], file_name)


    # gen_average = []
    # for i in range(k):
    #     t = np.average(gen_data[i, :])
    #     gen_average.append(t)
    #
    # # plot for generated data
    # plot_gen_data(real_data, gen_average, file_name)

    name = file_name.split("/")[2].split(".")[0]
    figure_dir = 'figure_real_gen_stl_' + str(iter) + name
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)

    k = 20
    real_data = random_sample(train_x, k)
    gen_data = gan.get_gen_out(k)[0]
    for i in range(k):
        plt.plot(real_data[i])
        plt.title('real_data point' + str(i))
        plt.savefig(figure_dir + '/real_' + str(i) + '.png')
        plt.close()
    for i in range(k):
        plt.plot(gen_data[i])
        plt.title('gen_data point' + str(i))
        plt.savefig(figure_dir + '/gen_' + str(i) + '.png')
        plt.close()


if __name__ == '__main__':

    # name = 'data/kpi/kpi_' + str(25) + '.csv'
    # run(name)

    # for j in range(20):
    #     for i in range(26):
    #         name = 'data/kpi/kpi_' + str(i) + '.csv'
    #         print('current file: ', name)
    #         run_1(name)
    #         tf.reset_default_graph()
    #         print()
    #         print('----------------')
    #     print('---------------------------------------------------------------------')
    #     print('---------------------------------------------------------------------')
    #     print('---------------------------------------------------------------------')
    #     print('---------------------------------------------------------------------')
    #     print()

    # os.rename('figure', 'figure-dim-50-label-stl')
    l = [24]
    for i in l:
        for j in range(10):
            name = 'data/kpi/kpi_' + str(i) + '.csv'
            print('current file: ', name)
            run(name, j)
            tf.reset_default_graph()
            print()
            print('----------------')

    # os.rename('figure', 'figure-2-dim-20')

    # for i in range(26):
    #     name = 'data/kpi/kpi_' + str(i) + '.csv'
    #     print('current file: ', name)
    #     run_1(name)
    #     tf.reset_default_graph()
    #     print()
    #     print('----------------')

    # l = [25, 11, 23, 24, 8, 8]
    # l = [24, 24, 24, 24, 24, 24]
    # for i in l:
    #     name = 'data/kpi/kpi_' + str(i) + '.csv'
    #     print('current file: ', name)
    #     run(name)
    #     tf.reset_default_graph()
    #     print()
    #     print('----------------')