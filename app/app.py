from flask import Flask, render_template, json, jsonify, request
#from _ast import IsNot
# from flask.ext.mysql import MySQL
# from werkzeug import generate_password_hash, check_password_hash

#mysql = MySQL()
app = Flask(__name__)

# MySQL configurations
#app.config['MYSQL_DATABASE_USER'] = 'jay'
#app.config['MYSQL_DATABASE_PASSWORD'] = 'jay'
#app.config['MYSQL_DATABASE_DB'] = 'BucketList'
#app.config['MYSQL_DATABASE_HOST'] = 'localhost'
#mysql.init_app(app)


@app.route('/')
def main():
    return render_template('index.html')

@app.route('/showSignUp')
def showSignUp():
    return render_template('signup.html')


@app.route('/analyze',methods=['POST','GET'])
def analyze():
    try:
        # POST request
        if request.method == 'POST':
            print('Incoming..')
            print(request.get_json())  # parse as JSON
            return 'OK', 200
    
        # GET request
        else:
            message = {'greeting':'Hello from Flask!'}
            return jsonify(message)  # serialize and use JSON headers

    except Exception as e:
        return json.dumps({'error':str(e)})
    finally:
        print("finally")
        #cursor.close() 
        #conn.close()

if __name__ == "__main__":
    app.run(port=5002)
