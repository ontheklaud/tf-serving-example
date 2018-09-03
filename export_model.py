from os import makedirs
import tensorflow as tf
from test import model


def main():

    # model config
    dtype = tf.float32
    model_name = 'model'
    model_version = 1

    len_X = 9
    len_Y = 1
    len_XY = len_X + len_Y

    # X/Y Placeholder and model
    X = tf.placeholder(dtype=dtype, shape=(None, len_X))
    Y = tf.placeholder(dtype=dtype, shape=(None, len_Y))
    net, loss = model(x=X, y=Y, in_size=len_X, out_size=len_Y)

    # session restore
    sess = tf.Session()
    saver = tf.train.Saver()
    saver_path = './model_ckpt/'
    last_ckpt_path = tf.train.latest_checkpoint(checkpoint_dir=saver_path)
    saver.restore(sess=sess, save_path=last_ckpt_path)

    # export builder
    builder = tf.saved_model.builder.SavedModelBuilder('{:s}/{:d}'.format(model_name, model_version))

    # tensor wrapping
    regression_input = tf.saved_model.utils.build_tensor_info(X)
    regression_output = tf.saved_model.utils.build_tensor_info(net)
    # regression_loss = tf.saved_model.utils.build_tensor_info(loss)

    # signature mapping
    regression_signatures = (
        tf.saved_model.signature_def_utils.build_signature_def(
            inputs=dict({
                tf.saved_model.signature_constants.REGRESS_INPUTS:
                    regression_input
            }),
            outputs=dict({
                tf.saved_model.signature_constants.REGRESS_OUTPUTS:
                    regression_output,
                # 'loss': regression_loss
            }),
            method_name=
                tf.saved_model.signature_constants.REGRESS_METHOD_NAME
        )
    )

    # prediction signature (?)
    tensor_info_x = tf.saved_model.utils.build_tensor_info(X)
    tensor_info_y = tf.saved_model.utils.build_tensor_info(net)

    prediction_signatures = tf.saved_model.signature_def_utils.build_signature_def(
        inputs={'x': tensor_info_x},
        outputs={'y': tensor_info_y},
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
    )

    legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')

    builder.add_meta_graph_and_variables(
        sess=sess, tags=[tf.saved_model.tag_constants.SERVING],
        signature_def_map=dict({
            'predict': prediction_signatures,
            'regression_value': regression_signatures
        }),
        legacy_init_op=legacy_init_op,
        # main_op = tf.tables_initializer(),
        strip_default_attrs=True,
    )

    builder.save()

    # fin
    return


if __name__ == '__main__':
    main()
