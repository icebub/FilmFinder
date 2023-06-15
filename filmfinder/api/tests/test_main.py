import unittest
from unittest.mock import patch

from api_core import load_config, load_model
from fastapi.testclient import TestClient
from main import app


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

            # Test with invalid input type still converts to string
            response = client.post(api_path, json={"text": 123})
            self.assertEqual(response.status_code, 200)

    def test_model_file(self):
        config = load_config()
        device = "cpu"

        with patch("api_core.load_transformer_model") as mock_load_transformer_model:
            mock_load_transformer_model.return_value = None, None
            model, tokenizer, reverse_mapping, thresholds, f1_mappping = load_model(
                config["exp_id"], device
            )

        self.assertGreater(len(f1_mappping.keys()), 0)
        print(f1_mappping[0])
        for label in f1_mappping.keys():
            for conf in range(1001):
                pred = str(format((conf / 1000), ".3f"))
                self.assertIsNotNone(f1_mappping[label][pred])
