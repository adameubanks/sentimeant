from flask import Flask, render_template, request, redirect, url_for
import stripe
import download

app = Flask(__name__)

pub_key = "pk_test_ZTlu8ZeGhxcvr013B9tXDcji008aGb26fG"
sec_key = "sk_test_NISIDuFJvjhYaRrSXF9Yn3gd006rKr2Xd8"

stripe.api_key = sec_key

@app.route('/')


def index():
    return render_template('home.html')

@app.route('/getstarted')
def form():
    return render_template('payment.html',pub_key=pub_key)

@app.route('/pay',methods=["POST"])
def pay():
    try:
        download.get_all_tweets(request.form.get('username', type=str))
    except TypeError:
        return render_template('form.html', message='invalid @')
    customer = stripe.Customer.create(email=request.form['stripeEmail'], source=request.form['stripeToken'])
    charge = stripe.Charge.create(customer=customer.id, amount=500, currency='usd',description='sentiment analysis')
    return redirect(url_for('diagnostic'))

@app.route('/diagnostic')
def diagnostic():
    return render_template('diagnostic.html')

if __name__ == "__main__":
    app.run(debug=True)
