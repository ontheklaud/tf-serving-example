from urllib.parse import urlencode
from urllib.request import Request, urlopen


def main():

    host = 'host'
    port = 8501
    version = 'v1'
    model_name = 'model'
    method = 'predict'

    file_POST = 'request.json'
    file_POST_block = open(file=file_POST, mode='rb').read()

    url = 'http://{:s}:{:d}/{:s}/models/{:s}:{:s}'.format(host, port, version, model_name, method)
    req = Request(url=url, data=file_POST_block)
    res = urlopen(url=req).read().decode()
    print(res)

    # fin
    return


if __name__ == '__main__':
    main()
