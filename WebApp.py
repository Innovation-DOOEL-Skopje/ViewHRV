# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import json
import os
import sys
from flask import Flask, request, jsonify, render_template, send_from_directory, flash, redirect, url_for, send_file
from flask_cors import CORS, cross_origin
from waitress import serve
from werkzeug.utils import secure_filename
from PIL import Image
from io import StringIO, BytesIO
from zipfile import ZipFile
import requests

import MitBih.MITDB_HRV_Calculate as MBH
import MitBih.Lombscargle_Plot as LOMB
import MitBih.Poincare_Plot as POIN
import MitBih.DFA_Plot as DFA
import MitBih.ECG_Plot as ECG
import MitBih.Utilities as Utilities

import matplotlib
import matplotlib.pyplot as plt

from Library.Calculate import Calculate as Calc
from Library.Statistical_Tools.Calculate_Time_Domain import Calculate_Time_Domain as ctd
from Library.Statistical_Tools.Calculate_Nonlinear_Measurements import Calculate_DFA_Features as cdfaf
from Library.Statistical_Tools.Calculate_Nonlinear_Measurements import Calculate_Poincare_Features as cpf
from Library.Statistical_Tools.Calculate_Frequency_Domain import Calculate_Lomb_Scargle as clsf

UPLOAD_FOLDER = './Uploads/'
ALLOWED_EXTENSIONS = {'txt', 'ann', 'csv', 'xlsx'}

plt.ioff()

app = Flask(__name__, static_folder="static")
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.debug = False
#app = Flask(__name__, static_url_path='/static')
# app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


#==============================================================================================================================================================================


# LANDING PAGE STARTS HERE
@app.route("/")
def main():
    return render_template('index.html')
# LANDING PAGE ENDS HERE

#==============================================================================================================================================================================
# MITBIH FUNCTIONALITIES START HERE

@app.route("/MitBihDB")
def MitBihDB():
    return render_template('MitBihDB.html')

cors = CORS(app, resources={r"/patient_id": {"origins": "*"}})

@app.route('/MitBihDB/Report',methods=["GET"])
@cross_origin(origin='*',headers=['Content- Type'])
def Calculate():
    # try:
        patient_id = request.args.get('patient_id')
        NN_Intervals = Utilities.Load_Data(patient_id)
        try:
            HRV= MBH.Calculate_HRV(patient_id)
        except:
            raise Exception('HRV Calcuation Failed')

        try:
            LOMB.plot_psd(NN_Intervals*1000, method="lomb")
        except:
            raise Exception('Lomb Failed')

        try:
            POIN.plot_poincare(NN_Intervals*1000)
        except:
            raise Exception('Poincare Failed')
        
        try:
            DFA.dfa(NN_Intervals, figsize=(10,10))
        except:
            raise Exception('DFA Failed')
        
        try:
            ECG.ECG(patient_id)
        except:
            raise Exception('ECG Failed')
        return render_template('Report.html', data=HRV)

    # except:
    #     return "Looks Like that patient doesnt exist"
    
    
# MITBIH FUNCTIONALITIES END HERE
#==============================================================================================================================================================================
# DEMO RENDERS START HERE

@app.route("/Demo")
def Demo():
    return render_template('Demo.html')

# DEMO RENDERS END HERE
#==============================================================================================================================================================================
# UPLOADING AND CALCULATING THE UPLOAD ENDPOINTS ARE LOCATED BELOW THIS LINE

@app.route("/Upload/")
def Upload():
    return render_template('Upload.html')

@app.route('/Results', methods = ['GET', 'POST'])
def Upload_File():
    f = request.files['file_txt']
    f.save("Uploads/File_ANN.ann")
    HRV = Calc("File_ANN", 128)

    return render_template('Results.html', data=HRV)

# END OF ENDPOINTS FOR UPLOADUING AND CALCULATING

# UNDER_CONSTRUCTION STARTS HERE

@app.route("/Under_Construction/")
def Under_Construction():
    return render_template('Under_Construction.html')

# UNDER_CONSTRUCTION ENDS 

#==============================================================================================================================================================================

# ERROR PAGES STARTS HERE

# Bad Request
@app.errorhandler(400)
def Page_400(e):
    return render_template('400.html'), 400

# Unathorized
@app.errorhandler(401)
def Page_401(e):
    return render_template('401.html'), 401

# Forbidden
@app.errorhandler(403)
def Page_403(e):
    return render_template('403.html'), 403

# Page Not Found
@app.errorhandler(404)
def Page_404(e):
    return render_template('404.html'), 404

# Internal Server Error
@app.errorhandler(500)
def Page_500(e):
    return render_template('500.html'), 500


# ERROR PAGES ENDS HERE

#==============================================================================================================================================================================

if __name__ == '__main__':
    app.run(debug = True, threaded=True)

# if __name__ == "__main__":
    #app.run() ##Replaced with below code to run it using waitress
#    serve(app, host='0.0.0.0', port=8000)