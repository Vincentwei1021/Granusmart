import os
from werkzeug.utils import secure_filename
# import gsmart_model as gs
import numpy

from flask import Flask, render_template, url_for, request, jsonify, redirect, send_from_directory

app = Flask(__name__)

aim_pic_path = ""
UPLOAD_FOLDER = 'static/uploads'  # 文件存放路径
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'png'])  # 限制上传文件格式

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/product')
def product():
    return render_template('product.html')


@app.route('/company')
def company():
    return render_template('company.html')


@app.route('/contact')
def contact():
    return render_template('contact.html')


@app.route('/upload', methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            relative_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(relative_path)
            return jsonify(status=1, relative_path=relative_path)
    return ''


@app.route('/run_model', methods=['GET'])
def run_model():
    input_pic_path = request.args.get('pic_path', type=str)
    print("Loading image:", input_pic_path)
    gs.load_image(input_pic_path)

    result = gs.run_model()

    print("Calculating results...")
    table_results = []
    table_results += list(gs.get_kernel_numbers(result))

    mask_chky = gs.get_chalky_mask(result)
    axes, kernel_length, kernel_width, kernel_length_to_width, _table = gs.get_kernel_ratio(result, table_results[0],
                                                                                            mask_chky)
    table_results += _table
    table_results = [num.item() for num in table_results]

    output_pic_path = 'static/img/dashboard.png'
    return jsonify(status=1, result_path=output_pic_path, table_results=table_results)


if __name__ == '__main__':
    app.run(host='0.0.0.0')
