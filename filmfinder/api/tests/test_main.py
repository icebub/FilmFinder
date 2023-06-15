import unittest
from unittest.mock import MagicMock, patch

from api_core import load_model, predict
from fastapi.testclient import TestClient
from main import app
from schema import BaseRequest, ResponseGenre


class TestPredict(unittest.TestCase):
    def test_predict_genre(self):
        with TestClient(app) as client:
            api_path = "/overview"

            sample_text = "Miles Morales catapults across the Multiverse, where he encounters a team of Spider-People charged with protecting its very existence. When the heroes clash on how to handle a new threat, Miles must redefine what it means to be a hero."
            response = client.post(api_path, json={"text": sample_text})
            self.assertEqual(response.status_code, 200)

            # Test with empty input
            response = client.post(api_path, json={"text": ""})
            self.assertEqual(response.status_code, 200)

            # Test with missing input
            response = client.post(api_path, json={})
            self.assertEqual(response.status_code, 422)

            # Test with input exceeding character limit
            sample_text = "a" * 10001
            response = client.post(api_path, json={"text": sample_text})
            self.assertEqual(response.status_code, 200)

            # Test with invalid input type still works
            response = client.post(api_path, json={"text": 123})
            self.assertEqual(response.status_code, 200)
