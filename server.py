import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from urllib.parse import parse_qs, urlparse
import json
import pandas as pd
from datetime import datetime
import uuid
import os
from typing import Callable, Any
from wsgiref.simple_server import make_server

nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)

adj_noun_pairs_count = {}
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

reviews = pd.read_csv('data/reviews.csv').to_dict('records')
filtered_locations = ["Albuquerque, New Mexico",
"Carlsbad, California",
"Chula Vista, California",
"Colorado Springs, Colorado",
"Denver, Colorado",
"El Cajon, California",
"El Paso, Texas",
"Escondido, California",
"Fresno, California",
"La Mesa, California",
"Las Vegas, Nevada",
"Los Angeles, California"
"Oceanside, California",
"Phoenix, Arizona",
"Sacramento, California"
"Salt Lake City, Utah",
"San Diego, California",
"Tucson, Arizona"]

class ReviewAnalyzerServer:
    def __init__(self) -> None:
        # This method is a placeholder for future initialization logic
        pass

    def analyze_sentiment(self, review_body):
        sentiment_scores = sia.polarity_scores(review_body)
        return sentiment_scores

    def __call__(self, environ: dict[str, Any], start_response: Callable[..., Any]) -> bytes:
        """
        The environ parameter is a dictionary containing some useful
        HTTP request information such as: REQUEST_METHOD, CONTENT_LENGTH, QUERY_STRING,
        PATH_INFO, CONTENT_TYPE, etc.
        """

        if environ["REQUEST_METHOD"] == "GET":
            final_output = []
            
            query = parse_qs(environ.get('QUERY_STRING',''))
            location_var = query.get('location',[None])[0]
            start_dt = query.get('start_date',[None])[0]
            end_dt = query.get('end_date',[None])[0]

            reviews_list = [i for i in reviews if((not location_var) or (location_var not in filtered_locations) or (i["Location"] == location_var)) and (not start_dt or i["Timestamp"] >= start_dt) and (not end_dt or i["Timestamp"] <= end_dt)]
            
            for review in reviews_list:
                review_copy = review.copy()
                sentiment = self.analyze_sentiment(review["ReviewBody"])
                review_copy["sentiment"] = sentiment
                final_output.append(review_copy)

            final_output = sorted(final_output, key=lambda i: i['sentiment']['compound'], reverse=True)

            # Create the response body from the reviews and convert to a JSON byte string
            response_body = json.dumps(final_output, indent=2).encode("utf-8")

            # Set the appropriate response headers
            start_response("200 OK", [
            ("Content-Type", "application/json"),
            ("Content-Length", str(len(response_body)))
             ])
            
            return [response_body]


        if environ["REQUEST_METHOD"] == "POST":
            query_stream = environ['wsgi.input'].read(int(environ.get('CONTENT_LENGTH',0)))
            query = parse_qs(query_stream.decode('utf-8'))

            location_var = query.get('Location',[None])[0]
            review_body = query.get('ReviewBody',[None])[0]

            if (not location_var) or (not review_body):
                start_response("400 Bad Request",[('Content-Type','application/json')])
                return [b'Invalid Location or ReviewBody']

            if location_var not in filtered_locations:
                start_response("400 Bad Request",[('Content-Type','application/json')])
                return [b'Invalid Location']
            
            timestamp_var = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            review_id_var = str(uuid.uuid4())

            review_obj = {
                "ReviewId": review_id_var,
                "ReviewBody": review_body,
                "Location": location_var,
                "Timestamp": timestamp_var
            }

            reviews.append(review_obj)

            response_body = json.dumps(review_obj).encode('utf-8')
            start_response("201 Created",[('Content-Type','application/json'),('Content-Length',str(len(response_body)))])
            return [response_body]

if __name__ == "__main__":
    app = ReviewAnalyzerServer()
    port = os.environ.get('PORT', 8000)
    with make_server("", port, app) as httpd:
        print(f"Listening on port {port}...")
        httpd.serve_forever()