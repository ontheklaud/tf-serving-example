# tf-serving-example
TensorFlow Serving Example

## Test Environemnt
```
python=3.6.3
tensorflow==1.10.0
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