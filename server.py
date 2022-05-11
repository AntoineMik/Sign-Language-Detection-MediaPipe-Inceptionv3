from helpers import *
from flask import Flask
from flask import jsonify
from flask import request
app = Flask(__name__)


@app.route("/random/images", methods=["POST"])
def raw():
    try:
        length = int(request.form.get('length'))
    except Exception as e:
        return jsonify({'message': 'Invalid request'}), 500
    if generate_user_raw(length):
        rand_img = display_img(user_test_path)
        rand_img.savefig("user_rand.jpg")
        result = jsonify({'data': "Success"})
    else:
        result = jsonify({'data': "failed"})

    return result


@app.route("/process/images", methods=["POST"])
def proc():
    try:
        length = int(request.form.get('length'))
    except Exception as e:
        return jsonify({'message': 'Invalid request'}), 500
    if generate_user_processed(length):
        rand_img = display_img(user_processed_path)
        rand_img.savefig("process_rand.jpg")
        result = jsonify({'data': "Success"})
    else:
        result = jsonify({'data': "failed"})

    return result


@app.route("/predict/images", methods=["POST"])
def pred():
    try:
        length = int(request.form.get('length'))
    except Exception as e:
        return jsonify({'message': 'Invalid request'}), 500

    prediction_img = predict_img(user_processed_path)
    if not prediction_img:
        result = jsonify({'Failed': "Generate random images before prediction"})
    else:
        prediction_img.savefig("user_pred.jpg")
        result = jsonify({'data': "Success"})

    return result


@app.route("/random", methods=["POST"])
def rand():
    try:
        length = int(request.form.get('length'))
    except Exception as e:
        return jsonify({'message': 'Invalid request'}), 500

    prediction_img = show_rand_pred(length, original_processed_path)
    prediction_img.savefig("rand_pred.jpg")
    result = jsonify({'data': "Success"})

    return result


@app.route("/", methods=["POST"])
def main():
    try:
        length = int(request.form.get('length'))
    except Exception as e:
        return jsonify({'message': 'Invalid request'}), 500

    result = jsonify({'Server': "Operational"})

    return result


if __name__ == "__main__":
    app.run(host="0.0.0.0", port="80")