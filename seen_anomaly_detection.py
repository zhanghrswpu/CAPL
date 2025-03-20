from __future__ import absolute_import
from __future__ import print_function
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, concatenate
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint
from keras import regularizers
from CA import CALayer
import argparse
from scipy.sparse import vstack, csc_matrix
from sklearn.model_selection import train_test_split
from utils_new import dataLoading, aucPerformance, writeResults, get_data_from_svmlight_file, dataLoading_noheader
from data_interpolation import inject_noise, inject_noise_sparse
import tensorflow as tf

MAX_INT = np.iinfo(np.int32).max
data_format = 0
ensemble_size = 30


def acr_classification_loss(y_true, y_pred):
    classification_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true, y_pred))
    return classification_loss


def pair_generator(x, outlier_indices, inlier_indices, Y, batch_size, nb_batch, rng):
    """batch generator
    """
    rng = np.random.RandomState(rng.randint(MAX_INT, size=1))
    counter = 0
    while 1:
        if data_format == 0:
            samples1, samples2, training_labels = pair_batch_generation(x, outlier_indices, inlier_indices, Y,
                                                                        batch_size, rng)
        counter += 1
        yield ([samples1, samples2], training_labels)
        if (counter > nb_batch):
            counter = 0


def pair_batch_generation(x_train, outlier_indices, inlier_indices, Y, batch_size, rng):
    dim = x_train.shape[1]
    pairs1 = np.empty((batch_size, dim))
    pairs2 = np.empty((batch_size, dim))
    labels = np.zeros((batch_size, 4))  # Modified to 4-dimensional labels
    n_inliers = len(inlier_indices)
    n_outliers = len(outlier_indices)

    block_size = int(batch_size / 5)

    # Normal + Normal sample pairs
    sid = rng.choice(n_inliers, block_size * 4, replace=False)
    pairs1[0:block_size * 2] = x_train[inlier_indices[sid[0:block_size * 2]]]
    pairs2[0:block_size * 2] = x_train[inlier_indices[sid[block_size * 2:block_size * 4]]]
    labels[0:block_size * 2, 0] = 1  # Normal + Normal

    # Normal + Anomalous sample pairs
    sid = rng.choice(n_inliers, block_size, replace=False)
    pairs1[block_size * 2:block_size * 3] = x_train[inlier_indices[sid]]
    sid = rng.choice(n_outliers, block_size)
    pairs2[block_size * 2:block_size * 3] = x_train[outlier_indices[sid]]
    labels[block_size * 2:block_size * 3, 1] = 1  # Normal + Anomalous

    # Anomalous + Normal sample pairs
    sid = rng.choice(n_outliers, block_size)
    pairs1[block_size * 3:block_size * 4] = x_train[outlier_indices[sid]]
    sid = rng.choice(n_inliers, block_size, replace=False)
    pairs2[block_size * 3:block_size * 4] = x_train[inlier_indices[sid]]
    labels[block_size * 3:block_size * 4, 2] = 1  # Anomalous + Normal

    # Anomalous + Anomalous sample pairs
    for i in np.arange(block_size * 4, batch_size):
        sid = rng.choice(n_outliers, 2, replace=False)
        z1 = x_train[outlier_indices[sid[0]]]
        z2 = x_train[outlier_indices[sid[1]]]
        pairs1[i] = z1
        pairs2[i] = z2
        labels[i, 3] = 1  # Anomalous + Anomalous

    return pairs1, pairs2, labels



def capl(input_shape, lambda_acr=1):
    x_input = Input(shape=input_shape)

    acr_layer = CALayer(lambda_acr=lambda_acr)
    centered_input = acr_layer(x_input)

    intermediate = Dense(20, activation='relu',
                         kernel_regularizer=regularizers.l2(h_lambda), name='hl1')(centered_input)
    base_network = Model(x_input, intermediate)

    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    input_merge = concatenate([processed_a, processed_b])

    output = Dense(4, activation='softmax', name='score')(input_merge)

    model = Model([input_a, input_b], output)
    model.compile(
        loss=lambda y_true, y_pred: acr_classification_loss(y_true, y_pred),
        optimizer=RMSprop(clipnorm=1.0)
    )

    return model


def ca_deeper(input_shape, lambda_acr=1.0):
    x_input = Input(shape=input_shape)

    acr_layer = CALayer(lambda_acr=lambda_acr)
    centered_input = acr_layer(x_input)

    intermediate = Dense(100, activation='relu',
                         kernel_regularizer=regularizers.l2(h_lambda), name='hl1')(centered_input)
    intermediate = Dense(20, activation='relu',
                         kernel_regularizer=regularizers.l2(h_lambda), name='hl2')(intermediate)
    base_network = Model(x_input, intermediate)
    print(base_network.summary())

    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    input_merge = concatenate([processed_a, processed_b])

    output = Dense(4, activation='softmax', name='score')(input_merge)

    model = Model([input_a, input_b], output)
    model.compile(
        loss=lambda y_true, y_pred: acr_classification_loss(y_true, y_pred),
        optimizer=RMSprop(clipnorm=1.0)
    )

    return model


def ca_no_feature_learner(input_shape):
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    input_merge = concatenate([input_a, input_b])
    output = Dense(4, activation='softmax', name='score')(input_merge)

    model = Model([input_a, input_b], output)
    model.compile(
        loss=lambda y_true, y_pred: acr_classification_loss(y_true, y_pred),
        optimizer=RMSprop(clipnorm=1.0)
    )

    return model


def load_model_weight_predict(model_name, input_shape, network_depth, x_test, inliers, outliers):
    if network_depth == 2:
        model = capl(input_shape, lambda_acr=1)
    elif network_depth == 1:
        model = ca_no_feature_learner(input_shape)
    else:
        model = ca_deeper(input_shape, lambda_acr=1)
    model.load_weights(model_name)
    scoring = Model(inputs=model.input, outputs=model.output)
    runs = ensemble_size
    rng = np.random.RandomState(42)
    test_size = x_test.shape[0]

    scores_00 = np.zeros((test_size, runs))  # Normal + Normal
    scores_01 = np.zeros((test_size, runs))  # Normal + Anomalous
    scores_10 = np.zeros((test_size, runs))  # Anomalous + Normal
    scores_11 = np.zeros((test_size, runs))  # Anomalous + Anomalous

    # Predict for Normal + Test
    n_sample = inliers.shape[0]
    for i in np.arange(runs):
        idx = rng.choice(n_sample, 1)
        obj = inliers[idx]
        inlier_seed = np.tile(obj, (test_size, 1))  # Normal samples
        predictions = scoring.predict([inlier_seed, x_test])
        scores_00[:, i] = predictions[:, 0]  # Normal + Normal channel
        scores_01[:, i] = predictions[:, 1]  # Normal + Anomalous channel

    # Predict for Test + Anomalous
    n_sample = outliers.shape[0]
    for i in np.arange(runs):
        idx = rng.choice(n_sample, 1)
        obj = outliers[idx]
        outlier_seed = np.tile(obj, (test_size, 1))  # Anomalous samples
        predictions = scoring.predict([outlier_seed, x_test])
        scores_10[:, i] = predictions[:, 2]  # Anomalous + Normal channel
        scores_11[:, i] = predictions[:, 3]  # Anomalous + Anomalous channel

    mean_score_00 = np.mean(scores_00, axis=1)
    mean_score_01 = np.mean(scores_01, axis=1)
    mean_score_10 = np.mean(scores_10, axis=1)
    mean_score_11 = np.mean(scores_11, axis=1)

    total_scores = np.zeros(test_size)

    for j in range(test_size):
        # Compute the sum of the four channel scores
        normal_sum = mean_score_00[j] + mean_score_10[j]  # Sum of channels 1 and 3 (Normal-related)
        anomaly_sum = mean_score_01[j] + mean_score_11[j]  # Sum of channels 2 and 4 (Anomalous-related)

        if normal_sum > anomaly_sum:
            total_scores[j] = 1 - normal_sum
        else:
            total_scores[j] = anomaly_sum

    return total_scores



def run_seen_anomaly_detection(args):
    names = 'UNSW_NB15_traintest_backdoor,UNSW_NB15_traintest_Reconnaissance,UNSW_NB15_traintest_DoS,UNSW_NB15_traintest_Fuzzers,celeba_baldvsnonbald_normalised,annthyroid_21feat_normalised,KDD2014_donors_10feat_nomissing_normalised'.split(
         ',')
    network_depth = int(args.network_depth)
    for nm in names:
        runs = args.runs
        rauc = np.zeros(runs)
        ap = np.zeros(runs)
        filename = nm.strip()
        n_outliers = 0
        global data_format
        data_format = int(args.data_format)
        if data_format == 0:
            x, labels = dataLoading(args.input_path + filename + ".csv")
        outlier_indices = np.where(labels == 1)[0]
        outliers = x[outlier_indices]
        n_outliers_org = outliers.shape[0]

        global h_lambda
        h_lambda = float(args.h_lambda)

        global ensemble_size
        ensemble_size = args.ensemble_size

        for i in np.arange(runs):
            x_train, x_test, y_train, y_test = train_test_split(x, labels, test_size=0.2, random_state=42,
                                                                stratify=labels)
            y_train = np.array(y_train)
            y_test = np.array(y_test)
            print(filename + ': round ' + str(i))
            outlier_indices = np.where(y_train == 1)[0]
            n_outliers = len(outlier_indices)
            print("Original training size: %d, No. outliers: %d" % (x_train.shape[0], n_outliers))

            n_noise = len(np.where(labels == 0)[0]) * args.cont_rate / (1. - args.cont_rate)
            n_noise = int(n_noise)

            rng = np.random.RandomState(42)
            if data_format == 0:
                if n_outliers > args.known_outliers:
                    mn = n_outliers - args.known_outliers
                    remove_idx = rng.choice(outlier_indices, mn, replace=False)
                    x_train = np.delete(x_train, remove_idx, axis=0)
                    y_train = np.delete(y_train, remove_idx, axis=0)

                noises = inject_noise(outliers, n_noise)
                x_train = np.append(x_train, noises, axis=0)
                y_train = np.append(y_train, np.zeros((noises.shape[0], 1)))

                noises = inject_noise_sparse(outliers, n_noise)
                x_train = vstack([x_train, noises])
                y_train = np.append(y_train, np.zeros((noises.shape[0], 1)))

            outlier_indices = np.where(y_train == 1)[0]
            inlier_indices = np.where(y_train == 0)[0]
            print(y_train.shape[0], outlier_indices.shape[0], inlier_indices.shape[0], n_noise)
            n_samples_trn = x_train.shape[0]
            input_shape = x_train.shape[1:]
            n_outliers = len(outlier_indices)
            print("Training data size: %d, No. outliers: %d" % (x_train.shape[0], n_outliers))
            Y = np.zeros(x_train.shape[0])
            Y[outlier_indices] = 1

            input_shape = x_train.shape[1:]
            epochs = args.epochs
            batch_size = args.batch_size
            nb_batch = args.nb_batch

            if network_depth == 2:
                model = capl(input_shape, lambda_acr=1)
            elif network_depth == 1:
                model = ca_no_feature_learner(input_shape)
            else:
                model = ca_deeper(input_shape, lambda_acr=1)

            model_name = "./model/capl_" + filename + "_" + str(args.cont_rate) + "cr_" + str(network_depth) + "d.h5"
            checkpointer = ModelCheckpoint(model_name, monitor='loss', verbose=0,
                                           save_best_only=True, save_weights_only=True)
            history = model.fit_generator(
                pair_generator(x_train, outlier_indices, inlier_indices, Y, batch_size, nb_batch, rng),
                steps_per_epoch=nb_batch,
                epochs=epochs,
                callbacks=[checkpointer])
            scores = load_model_weight_predict(model_name, input_shape, network_depth,
                                               x_test, x_train[inlier_indices], x_train[outlier_indices])
            rauc[i], ap[i] = aucPerformance(scores, y_test)

        mean_auc = np.mean(rauc)
        std_auc = np.std(rauc)
        mean_aucpr = np.mean(ap)
        std_aucpr = np.std(ap)
        print("average AUC-ROC: %.4f, average AUC-PR: %.4f" % (mean_auc, mean_aucpr))
        writeResults(filename + '_' + str(network_depth), x.shape[0], x.shape[1], mean_auc, mean_aucpr, std_auc, std_aucpr, path=args.output)


parser = argparse.ArgumentParser()
parser.add_argument("--network_depth", choices=['1', '2', '4'], default='2',
                    help="the depth of the network architecture")
parser.add_argument("--batch_size", type=int, default=512, help="batch size used in SGD")
parser.add_argument("--nb_batch", type=int, default=20, help="the number of batches per epoch")
parser.add_argument("--epochs", type=int, default=100, help="the number of epochs")
parser.add_argument("--runs", type=int, default=3,
                    help="how many times we repeat the experiments to obtain the average performance")
parser.add_argument("--known_outliers", type=int, default=60, help="the number of labeled outliers available at hand")
parser.add_argument("--cont_rate", type=float, default=0.02, help="the outlier contamination rate in the training data")
parser.add_argument("--ensemble_size", type=int, default=1,
                    help="ensemble_size. Using a size of one runs much faster while being able to obtain similarly good performance as using a size of 30.")
parser.add_argument("--h_lambda", type=float, default=0.005, help="regularization parameter")
parser.add_argument("--input_path", type=str, default='data/', help="the path of the data sets")
parser.add_argument("--data_format", choices=['0'], default='0',)
parser.add_argument("--output", type=str,
                    default='./results/seen_CAPL' + '.csv',
                    help="the output file path")

args = parser.parse_args()

run_seen_anomaly_detection(args)
