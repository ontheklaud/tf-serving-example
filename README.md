# tf-serving-example
TensorFlow Serving Example

## Test Environemnt
```
python=3.6.3
tensorflow==1.10.0
tensorflow-serving-api==1.10.1
```

## Filtering parameters while restoring model from checkpoint
```
# MUST initialize all parameters before restoring parameters
init_op = tf.global_variables_intializer()
sess.run(init_op)

f_list = ['keys_to_filter_1', 'keys_to_filter_2']

# g_col_ops is all unfiltered collection of vars & g_col_filtered_ops is all filtered collection of vars
g_col_ops, g_col_filtered_ops = get_graph_col_dict_filtered(f_list=f_list)

# initialize tensorflow checkpoint saver with UNFILTERD variables (g_col_ops)
ckpt_dir = 'checkpoint directory'
saver = tf.train.Saver(var_list=g_col_ops, ...)
latest_path = tf.train.latest_checkpoint(checkpoint_dir=ckpt_dir)

# restore checkpoint
saver.restore(sess=sess, save_path=latest_path)
```

## Test serving over RESTful API with JSON data
```
curl -d @request.json -X POST http://<Server>:<Port>/v1/models/model:predict
```

## Test serving over RESTful API with Client (cli_restful.py)
```
# edit client file - as you want!
python cli_restful.py
```

## Test serving over gRPC Protocol with Client (cli_grpc.py)
```
python cli_grpc.py --server host:port
```

## References
- https://github.com/tensorflow/serving
- https://www.tensorflow.org/serving/serving_basic
- https://www.tensorflow.org/serving/api_rest
- https://github.com/tensorflow/serving/blob/master/tensorflow_serving/g3doc/setup.md
- https://github.com/tensorflow/serving/blob/master/tensorflow_serving/g3doc/docker.md
- https://github.com/tensorflow/serving/blob/master/tensorflow_serving/example/mnist_saved_model.py
- https://docs.ksyun.com/read/latest/145/_book/deploy.html
- https://github.com/tensorflow/serving/blob/master/tensorflow_serving/g3doc/api_rest.md
- https://github.com/tensorflow/serving/tree/master/tensorflow_serving/servables/tensorflow/testdata
- https://raw.githubusercontent.com/tensorflow/serving/master/tensorflow_serving/example/mnist_client.py
- https://github.com/tensorflow/serving/blob/master/tensorflow_serving/example/mnist_client.py
