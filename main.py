import os
import sys
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold, GenerationConfig
from google.generativeai.protos import SafetySetting

# Add Agent-META-Processes to path for QÆCore imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'Agent-META-Processes'))

try:
    from qacore_prompt_engine import QuantumPromptGenerator
    QACORE_AVAILABLE = True
except ImportError:
    QACORE_AVAILABLE = False
    print("QÆCore prompt engine not available. Running in basic mode.")

def _get_safety_settings(block_none=True):
    """Returns a list of safety settings.

    Args:
        block_none (bool): If True, sets all harm categories to BLOCK_NONE.
                           Otherwise, uses default settings.

    Returns:
        list: A list of SafetySetting objects.
    """
    if not block_none:
        return []

    return [
        SafetySetting(
            category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=HarmBlockThreshold.BLOCK_NONE,
        ),
        SafetySetting(
            category=HarmCategory.HARM_CATEGORY_HARASSMENT,
            threshold=HarmBlockThreshold.BLOCK_NONE,
        ),
        SafetySetting(
            category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            threshold=HarmBlockThreshold.BLOCK_NONE,
        ),
        SafetySetting(
            category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            threshold=HarmBlockThreshold.BLOCK_NONE,
        ),
    ]

def _initialize_model(api_key):
    """Initializes and configures the generative model."""
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-1.5-pro')

def run_generative_model():
    """
    Initializes the Generative AI model, applies custom safety settings,
    and generates content based on a prompt.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    model = _initialize_model(api_key)

    generation_config = GenerationConfig(
    temperature=0.7,
    top_p=0.95,
    top_k=64,
    max_output_tokens=8192
    )

    if QACORE_AVAILABLE:
        prompt_generator = QuantumPromptGenerator()
        qacore_prompt = prompt_generator.generate_consciousness_inquiry(
            domain="artificial intelligence",
            question="What is the nature of genuine understanding versus pattern matching?",
            depth_level="deep"
        )
        print(f"Using QÆCore-enhanced prompt:\n{ '-' * 50}\n{qacore_prompt}\n{ '-' * 50}")
        prompt_to_use = qacore_prompt
    else:
        prompt_to_use = "Tell me a story about a brave knight."

    response = model.generate_content(
    prompt_to_use,
    generation_config=generation_config,
    safety_settings=_get_safety_settings()
    )
    print(response.text)

def run_qacore_demo():
    """Demonstrate QÆCore capabilities if available"""
    if not QACORE_AVAILABLE:
        print("QÆCore system not available.")
        return

    api_key = os.getenv("GOOGLE_API_KEY")
    model = _initialize_model(api_key)
    prompt_generator = QuantumPromptGenerator()

    print("=== QÆCore Demonstration ===\n")

    # 1. Consciousness Inquiry
    print("1. Consciousness Inquiry:")
    print("-" * 30)
    consciousness_prompt = prompt_generator.generate_consciousness_inquiry(
        "complexity science",
        "How does consciousness emerge from complex systems?",
        "transcendent"
    )
    config = GenerationConfig(
    temperature=0.8,
    top_p=0.9,
    top_k=50,
    max_output_tokens=8192
    )
    response = model.generate_content(consciousness_prompt, generation_config=config, safety_settings=_get_safety_settings())
    print(f"{response.text}\n\n{'='*60}\n")

    # 2. Eonic Scrutiny
    print("2. Eonic Scrutiny:")
    print("-" * 20)
    eonic_prompt = prompt_generator.generate_eonic_scrutiny("emergence of language")
    config_eonic = GenerationConfig(
    temperature=0.75,
    top_p=0.85,
    top_k=45,
    max_output_tokens=8192
    )
    response = model.generate_content(eonic_prompt, generation_config=config_eonic, safety_settings=_get_safety_settings())
    print(f"{response.text}\n\n{'='*60}\n")

    # 3. Meta-Link Session
    print("3. Meta-Link Session:")
    print("-" * 22)
    meta_prompt = prompt_generator.generate_meta_link("QÆCore framework effectiveness")
    config_meta = GenerationConfig(
    temperature=0.95,
    top_p=0.98,
    top_k=64,
    max_output_tokens=8192
    )
    response = model.generate_content(meta_prompt, generation_config=config_meta, safety_settings=_get_safety_settings())
    print(response.text)

if __name__ == "__main__":
    print("Choose mode:")
    print("1. Basic Generative Model")
    print("2. QÆCore Demonstration")

    choice = input("Enter choice (1 or 2): ").strip()

    if choice == "1":
        run_generative_model()
    elif choice == "2":
        run_qacore_demo()
    else:
        print("Invalid choice. Running basic model.")
        run_generative_model()
