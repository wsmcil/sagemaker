from __future__ import print_function
from flask import Flask, render_template, jsonify, redirect, request
import sagemaker
from sagemaker.mxnet import MXNetPredictor
from sagemaker.tensorflow import TensorFlowPredictor
from sagemaker.pytorch import PyTorchPredictor
import sys
import ast
import boto3
from boto3.dynamodb.conditions import Key, Attr
import decimal
import json
import datetime
from itertools import islice
import math
import struct
import io
import numpy

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/mxnet')
def mxnet():
    
    mynumber = request.args.getlist('image')

    predictor = MXNetPredictor('sagemaker-mxnet-2018-10-30-23-10-24-575')
    
    mynumberarray = ast.literal_eval(mynumber[0])
    
    response = predictor.predict(mynumberarray)
    
    labeled_predictions = list(zip(range(10), response[0]))

    labeled_predictions.sort(key=lambda label_and_prob: 1.0 - label_and_prob[1])
    answer= "Most likely answer: "+str(labeled_predictions[0])
    
    return(answer)

@app.route('/tensor')
def tensor():
    
    mynumber = request.args.getlist('image')

    predictor = TensorFlowPredictor('sagemaker-tensorflow-2018-10-31-23-19-00-978')
    
    mynumberarray = ast.literal_eval(mynumber[0])
    
    response = predictor.predict(mynumberarray)
    
    prediction = response['outputs']['classes']['int64_val'][0]
    
    answer= ("Most likely answer: {}".format(prediction))
    
    return(answer)

@app.route('/pytorch')
def pytorch():
    
    
#    import numpy as np

#image = np.array([data], dtype=np.float32)
#response = predictor.predict(image)
#prediction = response.argmax(axis=1)[0]
#print(prediction)
    
    
    mynumber = request.args.getlist('image')

    predictor = PyTorchPredictor('sagemaker-pytorch-2018-11-01-20-32-35-238')
    
    mynumberarray = np.array([mynumber], dtype=np.float32)
    #ast.literal_eval(mynumber[0])
    
    response = predictor.predict(mynumberarray)
    
    prediction = response['outputs']['classes']['int64_val'][0]
    
    answer= ("Most likely answer: {}".format(prediction))
    
    return(answer)
    
#    def np2csv(arr):
#        csv = io.BytesIO()
#        numpy.savetxt(csv, arr, delimiter=',', fmt='%g')
#        return csv.getvalue().decode().rstrip()
#    
#    mynumber = request.args.getlist('image')
#    
#    mynumberarray = ast.literal_eval(mynumber[0])
    #payload=mynumberarray
    #sagemaker-pytorch-2018-11-01-20-32-35-238
    
#    payload = np2csv(mynumberarray)
    
#    runtime_client = boto3.Session().client('runtime.sagemaker')
    
#    import json

#file_name = 'mnist.single.test' #customize to your test file 'mnist.single.test' if use the data above

#with open(file_name, 'r') as f:
#    payload = f.read()

#    endpoint_name='DEMO-XGBoostEndpoint-2018-11-01-18-44-19'

#    response = runtime_client.invoke_endpoint(EndpointName=endpoint_name, 
#                                              ContentType='text/x-libsvm', 
#                                              Body=payload)
    
    #print('Predicted label is {}.'.format(result))
    
    #response = runtime.invoke_endpoint(EndpointName='DEMO-XGBoostEndpoint-2018-04-20-03-07-34', 
    #                               ContentType='text/csv', 
    #                               Body=payload)
    
    #response = runtime.invoke_endpoint(EndpointName='DEMO-XGBoostEndpoint-2018-11-01-18-44-19', 
    #                               ContentType='text/csv', 
    #                               Body=payload)
    
#   result = response['Body'].read().decode('ascii')
#    floatArr = numpy.array(json.loads(result))
#    predictedLabel = numpy.argmax(floatArr)
#    
#    answer= ("Most likely answer: {}".format(predictedLabel))
    
#    return(answer)

@app.route('/save')
def save():

    # Helper class to convert a DynamoDB item to JSON.
    class DecimalEncoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, decimal.Decimal):
                if o % 1 > 0:
                    return float(o)
                else:
                    return int(o)
            return super(DecimalEncoder, self).default(o)

    dynamodb = boto3.resource('dynamodb', region_name='us-west-2', endpoint_url="https://dynamodb.us-west-2.amazonaws.com")

    table = dynamodb.Table('ml-game-observations')

    algorithm = request.args.getlist('algorithm')
    actual=request.args.getlist('actual')
    guess=request.args.getlist('guess')
    image=request.args.getlist('image')
    currentDT = datetime.datetime.now()

    response = table.put_item(
       Item={
            'datetime': str(currentDT),
            'info': {
                'algorithm': algorithm,
                'actual': actual,
                'guess': guess,
                'image': image
            }
        }
    )
    
    return("saved")
    
if __name__ == "__main__":
    app.run(debug=False)