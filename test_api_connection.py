"""
Quick API Connection Test
===========================
Tests if ALICE API credentials are working correctly.
"""

import asyncio
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI

async def test_api():
    """Test API connection with a simple query"""
    
    # Load environment variables
    load_dotenv()
    
    api_key = os.getenv('ALICE_OPENAI_KEY')
    base_url = os.getenv('ALICE_CHAT_URL')
    
    print("="*80)
    print("API Connection Test")
    print("="*80)
    print(f"API Key: {api_key[:20]}...{api_key[-10:] if api_key else 'NOT FOUND'}")
    print(f"Base URL: {base_url}")
    print()
    
    if not api_key or not base_url:
        print("ERROR: Environment variables not loaded!")
        print("ALICE_OPENAI_KEY:", "Found" if api_key else "NOT FOUND")
        print("ALICE_CHAT_URL:", "Found" if base_url else "NOT FOUND")
        return False
    
    try:
        # Initialize client
        client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url
        )
        
        print("Testing simple API call...")
        print()
        
        # Simple test message
        response = await client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {"role": "user", "content": "Say 'API connection successful!' and nothing else."}
            ],
            temperature=0.0
        )
        
        result = response.choices[0].message.content
        
        print("Response:", result)
        print()
        
        if "successful" in result.lower() or "success" in result.lower():
            print("✓ API Connection Test PASSED")
            return True
        else:
            print("✓ API responded but unexpected message")
            return True
            
    except Exception as e:
        print(f"✗ API Connection Test FAILED")
        print(f"Error: {str(e)}")
        print()
        print("Error type:", type(e).__name__)
        return False

if __name__ == "__main__":
    success = asyncio.run(test_api())
    exit(0 if success else 1)
