import os
TIMEZONE = 'Europe/Paris'

# Secret key for generating tokens
SECRET_KEY = 'houdini'

# Admin credentials
ADMIN_CREDENTIALS = ('admin', 'pa$$word')

# Database choice
SQLALCHEMY_DATABASE_URI = 'sqlite:///app.db'
SQLALCHEMY_TRACK_MODIFICATIONS = True

# Configuration of a Gmail account for sending mails
MAIL_SERVER = 'smtp.googlemail.com'
MAIL_PORT = 465
MAIL_USE_TLS = False
MAIL_USE_SSL = True
MAIL_USERNAME = 'flask.boilerplate'
MAIL_PASSWORD = 'flaskboilerplate123'
ADMINS = ['flask.boilerplate@gmail.com']

# Number of times a password is hashed
BCRYPT_LOG_ROUNDS = 12

#PATH = '/opt/render/project/src' # This is for a render deployment, change to your path if you are running on localhost
#UPLOAD_FOLDER = '/opt/render/project/src/app/static'
#MODELS_PATH = '/opt/render/project/src/models/baseline.npz'
# Deployment to AWS
PATH = os.path.abspath(os.path.join(os.getcwd(),os.pardir)) # This is for a render deployment, change to your path if you are running on localhost
UPLOAD_FOLDER = os.getcwd() + '/static'
MODELS_PATH = PATH + '/models/baseline.npz'
