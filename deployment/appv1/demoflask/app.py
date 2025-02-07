from flask import Flask, render_template, request


app = Flask(__name__)

@app.route('/')
def welcome():
    return "Welcome to the Flask App!"


@app.route('/index')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)