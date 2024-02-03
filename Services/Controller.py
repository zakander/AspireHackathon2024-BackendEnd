from flask import Flask, request, jsonify
import requests
import json

from Services.AgentService.AgentService import AgentService

app = Flask(__name__)

from flask import Flask, request

app = Flask(__name__)

langchain_url = 'https://api.langchain.com/chat'
webhook_url = 'https://webhook.site/9c4c73ae-a16d-4c0a-9987-e170e110256c'

class Controller:
    def __init__(self):
        self.AgentService = AgentService()
    
    @app.route('/lps-webhook', methods=['POST'])
    def send_prediction_data():
        # Input data from HTTP
        response_data = request.json
        prediction_data = AgentService.predict(response_data)
        response = requests.post(
            # langchain_url,
            json={
                'text': 'Learning Service Predictions',
                'webhook': webhook_url,
                'data': prediction_data
            }
        )
        return 'OK'