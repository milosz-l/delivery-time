from flask import Flask, request

app = Flask(__name__)

@app.route('/', methods=['POST'])
def result():
    print("NEW POST REQUEST")
    data = request.json
    print("DATA", data)  # json (if content-type of application/json is sent with the request)
    return "OK"