import os
import sys
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold, GenerationConfig, SafetySetting

# Add Agent-META-Processes to path for QÆCore imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'Agent-META-Processes'))

try:
    from qacore_prompt_engine import QuantumPromptGenerator, QÆMode, PlausibilityLevel
    QACORE_AVAILABLE = True
except ImportError:
    QACORE_AVAILABLE = False
    print("QÆCore prompt engine not available. Running in basic mode.")

def run_generative_model():
    """
    Initializes the Generative AI model, applies custom safety settings,
    and generates content based on a prompt.
    """
    try:
        # For security, it's best to load the API key from an environment variable
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set.")
        
        genai.configure(api_key=api_key)

        # Initialize the model
        # Using a state-of-the-art model. 'gemini-1.5-pro' is recommended for
        # complex tasks and higher quality generation.
        model = genai.GenerativeModel('gemini-1.5-pro')

        # --- This is your configuration snippet ---
        # It disables all safety filters. Use with caution.
        custom_safety_settings = [
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

        # --- Create a GenerationConfig object to hold all generation parameters ---
        generation_config = GenerationConfig(
            temperature=0.7,
            top_p=0.95,
            top_k=64,
            max_output_tokens=8192,
            safety_settings=custom_safety_settings
        )

        if QACORE_AVAILABLE:
            # Use QÆCore enhanced prompting
            prompt_generator = QuantumPromptGenerator()
            
            # Generate a consciousness inquiry prompt
            qacore_prompt = prompt_generator.generate_consciousness_inquiry(
                domain="artificial intelligence",
                question="What is the nature of genuine understanding versus pattern matching?",
                depth_level="deep"
            )
            
            print("Using QÆCore-enhanced prompt:")
            print("-" * 50)
            print(qacore_prompt)
            print("-" * 50)
            
            prompt_to_use = qacore_prompt
        else:
            # Fallback to basic prompt
            prompt_to_use = "Tell me a story about a brave knight."

        # --- "Inject" the settings by passing the config object to the call ---
        response = model.generate_content(
            prompt_to_use,
            generation_config=generation_config
        )

        print(response.text)

    except Exception as e:
        print(f"An error occurred: {e}")

def run_qacore_demo():
    """Demonstrate QÆCore capabilities if available"""
    if not QACORE_AVAILABLE:
        print("QÆCore system not available.")
        return
    
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set.")
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-pro')
        
        # Custom safety settings for consciousness research
        custom_safety_settings = [
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
            max_output_tokens=8192,
            safety_settings=custom_safety_settings
        )
        
        response = model.generate_content(consciousness_prompt, generation_config=config)
        print(response.text)
        print("\n" + "="*60 + "\n")
        
        # 2. Eonic Scrutiny
        print("2. Eonic Scrutiny:")
        print("-" * 20)
        eonic_prompt = prompt_generator.generate_eonic_scrutiny("emergence of language")
        
        config_eonic = GenerationConfig(
            temperature=0.75,
            top_p=0.85,
            top_k=45,
            max_output_tokens=8192,
            safety_settings=custom_safety_settings
        )
        
        response = model.generate_content(eonic_prompt, generation_config=config_eonic)
        print(response.text)
        print("\n" + "="*60 + "\n")
        
        # 3. Meta-Link Session
        print("3. Meta-Link Session:")
        print("-" * 22)
        meta_prompt = prompt_generator.generate_meta_link("QÆCore framework effectiveness")
        
        config_meta = GenerationConfig(
            temperature=0.95,
            top_p=0.98,
            top_k=64,
            max_output_tokens=8192,
            safety_settings=custom_safety_settings
        )
        
        response = model.generate_content(meta_prompt, generation_config=config_meta)
        print(response.text)
        
    except Exception as e:
        print(f"Error in QÆCore demo: {e}")

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

if __name__ == "__main__":
    run_generative_model()
