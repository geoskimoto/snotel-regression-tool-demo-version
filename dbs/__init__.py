# -*- coding: utf-8 -*-
import os
from pathlib import Path
from datetime import datetime
import dash
from auth import basic_auth
from flask_sqlalchemy import SQLAlchemy
import dash_bootstrap_components as dbc

DB_DIR = os.path.dirname(os.path.realpath(__file__))
APP_DIR = os.path.dirname(DB_DIR)
APP_TITLE = "SNOTEL Regression Tool - Beta"
USE_AUTH = os.getenv("USE_AUTH", False)
AUTH_USER = os.getenv("AUTH_USER", "user")
AUTH_PASSWORD = os.getenv("AUTH_PASSWORD", "snotel")


def create_app(use_auth=USE_AUTH):
    assets_path = Path(APP_DIR, "assets")
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.YETI],
        update_title="Updating...",
        suppress_callback_exceptions=True,
        assets_folder=assets_path,
    )
    app.title = APP_TITLE
    if use_auth:
        print(f"Using basic auth - env var USE_AUTH = {use_auth}")
        basic_auth.BasicAuth(app, {AUTH_USER: AUTH_PASSWORD})
    meta_db_path = Path(DB_DIR, "meta.db")
    models_db_path = Path(DB_DIR, 'regr_models.db')
    meta_db_con_str = f"sqlite:///{meta_db_path.as_posix()}"
    models_db_con_str = f"sqlite:///{models_db_path.as_posix()}"
    app.server.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.server.config["SQLALCHEMY_DATABASE_URI"] = meta_db_con_str
    app.server.config['SQLALCHEMY_BINDS'] = {
        'regr_models': models_db_con_str,
    }


    return app


app = create_app()
db = SQLAlchemy(app.server)
# db.reflect()

#Make sure to delete .db file each time you make a change!

class Regression_Models(db.Model):
    __bind_key__ = 'regr_models'
    __tablename__ = 'regression_models'
    id = db.Column(db.Integer, primary_key = True)
    response = db.Column(db.String(50))  #Can't set to primary_key as each key has to be unique, meaning if reponse was set as primary, there could only be one saved model for each station.
    # predictors = db.Column(db.String(200))
    stationparameters = db.Column(db.String())
    sdate = db.Column(db.String())
    edate = db.Column(db.String())
    regressor_type = db.Column(db.String())    
    RMSE_train = db.Column(db.Integer)
    RMSE_test = db.Column(db.Integer)
    model = db.Column(db.String()) #db.Column(db.PickleType)  #This is throwing a JSON serialization error when trying to redisplay the saved model in a dash_table since it's a pickled file.  
                                                              #Will need to go in and convert the dictionary that's created from the df that feeds the table and change 
                                                              # it manually to a string just for the table and not actually in the db.
    date_created = db.Column(db.DateTime, default=datetime.now())
    

    def __init__(self, response, stationparameters, sdate, edate, regressor_type, RMSE_train, RMSE_test, model): #, date_created):
        self.response = response
        # self.predictors = predictors
        self.stationparameters = stationparameters 
        self.sdate = sdate
        self.edate = edate
        self.regressor_type = regressor_type       
        self.RMSE_train = RMSE_train
        self.RMSE_test = RMSE_test  
        self.model = model


    
# if os.path.isfile('./regr_models.db'):
#     pass
# else:
with app.server.app_context():
    db.create_all()
    db.session.commit()


        

if __name__ == "__main__":
    if os.path.isfile('./regr_models.db'):
        pass
    else:
        db.create_all()
        
 #   # tbl = "Regression_Models"
  #  # df = pd.read_sql(tbl, con=db.get_engine())
   # # print(df.tail())
    
    print('Creates new regr_models.db if it does not already exist')
