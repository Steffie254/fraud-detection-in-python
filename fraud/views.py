import datetime
from datetime import datetime

from django.shortcuts import render
import os
import pickle
import numpy as np
import pandas as pd
from sklearn import datasets
from django.conf import settings
from rest_framework import views
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

# Create your views here.

from django.http import JsonResponse
from rest_framework.decorators import api_view
import pickle


class PredictAPIView(APIView):
    def post(self, request):
        try:
            age = request.data.get("Age")
            gender = request.data.get("Gender")
            amount = request.data.get("Amount")
            merchant_category = request.data.get("Gender")


        # Load the machine learning model from the pickle file
            with open('pickle/fraud_detector.pkl', 'rb') as model_file:
                model = pickle.load(model_file)

            # Get the input data from the request
                input_data = request.data

                required_features = ["age", "amount", "M", "es_barsandrestaurants", "es_contents", 
                                     "es_fashion", "es_food", "es_health", "es_home", "es_hotelservices", "es_hyper", "es_leisure", "es_otherservices", "es_sportsandtoys", "es_tech", "es_transportation", "es_travel"]

                for feature in required_features:
                    if feature not in input_data:
                        raise ValueError(f"Missing '{feature}' in input data.")

            # Prepare the input data as a list for prediction
                input_values = [input_data[feature] for feature in required_features]

                numeric_fields = ["age", "amount"]
                for field in numeric_fields:
                    if field in input_data:
                        input_data[field] = float(input_data[field])


        # Perform prediction using the loaded model[]
                print ("input_values", input_values)
                prediction = model.predict([input_values])
                
        # Return the prediction as JSON response
                return JsonResponse({'prediction': prediction.tolist()})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)


class PredictRanksAPIView(APIView):
    def post(self, request):
        try:
            # Get the input data from the request
            #age = request.data.get("Age")
            gender = request.data.get("Gender")
            amount = float(request.data.get("Amount")) 
            merchant_category = request.data.get("MerchantCategory")

            # AGE
            # Get the account creation date (assuming it's provided in the request)
            account_created_at_str = "10/11/2023" # You should adjust this to match the actual key in your request
            account_created_at = datetime.strptime(account_created_at_str, "%m/%d/%Y")


            # Calculate the time difference between the current date and account creation date
            # current_date = datetime.datetime.today()
            current_date = datetime.now()
            account_age = current_date - account_created_at


            # Check if the account age is less than 1 day
            # is_new_account = account_age.total_seconds() < 24 * 60 * 60  # 24 hours in seconds
            is_new_account = int(account_age.total_seconds() < 24 * 60 * 60)

            # GENDER
            is_male = 1 if gender == "M" else 0
            
            # AMOUNT 
              # Check if the amount is more than 1,000,000p
            is_large_transaction = int(amount > 1000000)

            # CATEGORIES
            # Include the merchant category flags as input features
            #listen to category, if presentoverride default
            merchant_category = [
                request.data.get("es_barsandrestaurants"),
                request.data.get("es_contents", 0),
                request.data.get("es_fashion", 0),
                request.data.get("es_food", 0),
                request.data.get("es_health", 0),
                request.data.get("es_home", 0),
                request.data.get("es_hotelservices", 1),
                request.data.get("es_hyper", 0),
                request.data.get("es_leisure", 0),
                request.data.get("es_otherservices", 0),
                request.data.get("es_sportsandtoys", 0),
                request.data.get("es_tech", 0),
                request.data.get("es_transportation", 0),
                request.data.get("es_travel", 0)
            ]

            # Prepare the input features for the machine learning model
            # input_values = [age, amount, is_male]

            input_values = [
                is_male, is_new_account, is_large_transaction
            ] + merchant_category

            # Load the machine learning model from the pickle file
            with open('pickle/fraud_detector.pkl', 'rb') as model_file:
                model = pickle.load(model_file)


        # Perform prediction using the loaded model[]
            print ("input_values", input_values)
            prediction = model.predict([input_values])


        # Return the prediction as JSON response
            return JsonResponse({'prediction': prediction.tolist()})
        
        except Exception as e:

            return JsonResponse({'error': str(e)}, status=400)
        
