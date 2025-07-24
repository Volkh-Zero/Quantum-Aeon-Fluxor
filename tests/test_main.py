
import unittest
from unittest.mock import patch, MagicMock
import os
import sys

# Add the parent directory to the path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import run_generative_model, run_qacore_demo

class TestMain(unittest.TestCase):

    @patch('main.genai')
    @patch('main.os.getenv')
    def test_run_generative_model_basic(self, mock_getenv, mock_genai):
        """Test the basic functionality of run_generative_model without QÆCore."""
        mock_getenv.return_value = "fake_api_key"
        
        # Mock the generative model and its response
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "This is a test story."
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model
        
        # Set QACORE_AVAILABLE to False to test the basic path
        with patch('main.QACORE_AVAILABLE', False):
            with patch('builtins.print') as mock_print:
                run_generative_model()
                
                # Check if the model was initialized and content generated
                mock_genai.configure.assert_called_with(api_key="fake_api_key")
                mock_genai.GenerativeModel.assert_called_with('gemini-1.5-pro')
                mock_model.generate_content.assert_called_once()
                
                # Check that the fallback prompt was used
                self.assertIn("Tell me a story about a brave knight.", mock_model.generate_content.call_args[0][0])
                
                # Check that the response was printed
                mock_print.assert_any_call("This is a test story.")

    @patch('main.genai')
    @patch('main.os.getenv')
    @patch('main.QuantumPromptGenerator')
    def test_run_generative_model_qacore(self, mock_prompt_generator, mock_getenv, mock_genai):
        """Test the QÆCore-enhanced functionality of run_generative_model."""
        mock_getenv.return_value = "fake_api_key"
        
        # Mock the generative model and its response
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "This is a QÆCore response."
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model
        
        # Mock the prompt generator
        mock_prompt_instance = MagicMock()
        mock_prompt_instance.generate_consciousness_inquiry.return_value = "QÆCore prompt"
        mock_prompt_generator.return_value = mock_prompt_instance
        
        # Set QACORE_AVAILABLE to True
        with patch('main.QACORE_AVAILABLE', True):
            with patch('builtins.print') as mock_print:
                run_generative_model()
                
                # Check that the QÆCore prompt was used
                mock_model.generate_content.assert_called_once_with(
                    "QÆCore prompt",
                    generation_config=unittest.mock.ANY,
                    safety_settings=unittest.mock.ANY
                )
                mock_print.assert_any_call("This is a QÆCore response.")

    @patch('main.genai')
    @patch('main.os.getenv')
    @patch('main.QuantumPromptGenerator')
    def test_run_qacore_demo(self, mock_prompt_generator, mock_getenv, mock_genai):
        """Test the QÆCore demonstration function."""
        mock_getenv.return_value = "fake_api_key"
        
        # Mock the generative model and its response
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "QÆCore demo text."
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model
        
        # Mock the prompt generator
        mock_prompt_instance = MagicMock()
        mock_prompt_instance.generate_consciousness_inquiry.return_value = "consciousness prompt"
        mock_prompt_instance.generate_eonic_scrutiny.return_value = "eonic prompt"
        mock_prompt_instance.generate_meta_link.return_value = "meta prompt"
        mock_prompt_generator.return_value = mock_prompt_instance
        
        with patch('main.QACORE_AVAILABLE', True):
            with patch('builtins.print'):
                run_qacore_demo()
                
                # Check that the model was called for each demo section
                self.assertEqual(mock_model.generate_content.call_count, 3)

if __name__ == '__main__':
    unittest.main()
