from flask import Flask, render_template, request


app = Flask(__name__)

# @app.route('/')
# def welcome():
#     return "Welcome to the Flask Appdsadasfsa fdsgdsgds dasdsad!"


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/register')
def register():
    return render_template('forms.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/services')
def services():
    return render_template('services.html')

if __name__ == '__main__':
    app.run(debug=True)