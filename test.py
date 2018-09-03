from functools import partial
from os import makedirs
import tensorflow as tf


def model(x=None, y=None, in_size=1, out_size=1):

    Weights = tf.Variable(tf.random_normal([in_size, out_size], mean=0., stddev=1.))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)

    # fully connected product
    Wx_plus_b = tf.matmul(x, Weights) + biases

    # mse (l2 loss)
    loss = tf.nn.l2_loss(y - Wx_plus_b)

    # fin
    return Wx_plus_b, loss


def map_func(len_X, len_Y, *args):

    # CSVDataset split each batch_input into column-wise
    # So, concatenation is predicated during dataset mapping

    # change shape into: batch x column
    XY_stack = tf.stack(values=args, axis=-1)

    # split X/Y for n_X:n_Y
    split_x, split_y = tf.split(value=XY_stack, num_or_size_splits=[len_X, len_Y], axis=-1)

    # debug info for shape validation
    # print(split_x, split_y)
    # feed_dict = dict({X: split_x, Y: split_y})
    # print('map_func:', XY_stack)

    # fin
    return split_x, split_y


def main():

    # model config
    dtype = tf.float32
    epoch = 10
    batch = 10
    fetch_buffer = 5
    opt_lr = 1e-4

    len_X = 1032
    len_Y = 2
    len_XY = len_X + len_Y

    # use CSV Dataset with... Random Shuffling (reset per repeat) & mini Batch & map
    dataset = tf.contrib.data.CsvDataset(
        filenames="sample.csv",
        record_defaults=[tf.float32 for _ in range(len_XY)],
        select_cols=list(range(0, len_X)))\
        .shuffle(buffer_size=batch*fetch_buffer)\
        .batch(batch_size=batch)\
        .map(map_func=partial(map_func, len_X, len_Y))

    # create initializable iterator from CSV Dataset
    dataset_iter = dataset.make_initializable_iterator()
    get_next_batch = dataset_iter.get_next()

    # X/Y Placeholder
    X = tf.placeholder(dtype=dtype, shape=(None, len_X))
    Y = tf.placeholder(dtype=dtype, shape=(None, len_Y))

    # NN model
    net, loss = model(x=X, y=Y, in_size=len_X, out_size=len_Y)

    # minimize loss
    global_step = tf.train.get_or_create_global_step()
    opt = tf.train.AdamOptimizer(learning_rate=opt_lr)

    # tensor ops
    loss_op = opt.minimize(loss, global_step=global_step)
    init_op = tf.global_variables_initializer()
    data_op = dataset_iter.initializer

    # sess_init
    sess = tf.Session()
    sess.run(init_op)

    # training demo
    for ep in range(epoch):

        # init (or re-init) dataset iterator per each iterator
        sess.run(data_op)

        while True:

            try:
                # render x/y values
                batch_x, batch_y = sess.run(get_next_batch)
                feed_dict = dict({X: batch_x, Y: batch_y})

                # feed redered x/y values into placeholder
                _, net_out, loss_out, g_step_out = sess.run([loss_op, net, loss, global_step], feed_dict=feed_dict)

                # print training step/loss
                print('[{:d}] loss = {:.4f}'.format(g_step_out, loss_out))

            except tf.errors.OutOfRangeError:
                # just finish this epoch

                print('ep {:d} done'.format(ep+1))
                break

    # save model graph
    saver = tf.train.Saver()
    makedirs('model', exist_ok=True)
    saver.save(sess=sess, save_path='model/model.ckpt', global_step=global_step)

    # fin
    return


if __name__ == '__main__':
    main()
