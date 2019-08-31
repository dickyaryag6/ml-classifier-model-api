from django.shortcuts import render

from .apps import ClassifiermodelConfig

# Create your views here.
from django.http import HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .apps import WebappConfig


from model import NLPModel()

model = NLPModel()

# clf_path = 'lib/models/SentimentClassifier.pkl'
# with open(clf_path, 'rb') as f:
#     model.clf = pickle.load(f)
#
# vec_path = 'lib/models/TFIDFVectorizer.pkl'
# with open(vec_path, 'rb') as f:
#     model.vectorizer = pickle.load(f)

# argument parsing
parser = reqparse.RequestParser()
parser.add_argument('query')

class call_model(APIView):

    def get(self,request):
        if request.method == 'GET':
                args = parser.parse_args()
                user_query = args['query']
                    # vectorize the user's query and make a prediction
                uq_vectorized = model.vectorizer_transform(
                    np.array([user_query]))
                prediction = model.predict(uq_vectorized)
                pred_proba = model.predict_proba(uq_vectorized)
                    # Output 'Negative' or 'Positive' along with the score
                if prediction == 0:
                    pred_text = 'Negative'
                else:
                    pred_text = 'Positive'

                    # round the predict proba value and set to new variable
                confidence = round(pred_proba[0], 3)
                    # create JSON object
                output = {'prediction': pred_text, 'confidence': confidence}

                return output

            # # sentence is the query we want to get the prediction for
            # params =  request.GET.get('sentence')
            #
            # # predict method used to get the prediction
            # response = ClassifiermodelConfig.predictor.predict(sentence)
            #
            # # returning JSON response
            # return JsonResponse(response)

# class PredictSentiment(Resource):
#     def get(self):
#         # use parser and find the user's query
#         args = parser.parse_args()
#         user_query = args['query']
#         # vectorize the user's query and make a prediction
#         uq_vectorized = model.vectorizer_transform(
#             np.array([user_query]))
#         prediction = model.predict(uq_vectorized)
#         pred_proba = model.predict_proba(uq_vectorized)
#         # Output 'Negative' or 'Positive' along with the score
#         if prediction == 0:
#             pred_text = 'Negative'
#         else:
#             pred_text = 'Positive'
#
#         # round the predict proba value and set to new variable
#         confidence = round(pred_proba[0], 3)
#         # create JSON object
#         output = {'prediction': pred_text, 'confidence': confidence}
#
#         return output
