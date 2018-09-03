# tf-serving-example
TensorFlow Serving Example

## Test Environemnt
```
python=3.6.3
tensorflow==1.10.0
tensorflow-serving-api==1.10.1
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
