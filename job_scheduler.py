import time
from flask import Flask
from flask_apscheduler import APScheduler

app = Flask(__name__)

scheduler= APScheduler()

def print_date_time():
    print(time.strftime("%A, %d. %B %Y %I:%M:%S %p"))



# Shut down the scheduler when exiting the app
if __name__ == '__main__':
    scheduler.add_job(func=print_date_time, trigger="interval", minutes=1, id="ok")
    scheduler.start() 
    app.run(host='0.0.0.0',debug=False)