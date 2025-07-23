import asyncio
import json
from openai import AsyncOpenAI

async def handle_streaming_response(response, description="Streaming response"):
    """Helper function to handle streaming responses consistently"""
    print(f"📤 {description}:")
    reasoning_content = ""
    final_content = ""
    reasoning_started = False
    content_started = False
    
    try:
        async for chunk in response:
            try:
                chunk_dict = chunk.model_dump()
                
                # Handle final chunk with complete message
                if chunk_dict.get("object") == "chat.completion":
                    message = chunk_dict["choices"][0].get("message", {})
                    if message.get("reasoning_content"):
                        print("\n🧠 Final Reasoning Content:")
                        print(message["reasoning_content"])
                    if message.get("content"):
                        print("\n💬 Final Response Content:")
                        print(message["content"])
                    break
                
                # Handle streaming chunks
                elif chunk_dict.get("object") == "chat.completion.chunk":
                    choices = chunk_dict.get("choices", [])
                    if not choices:
                        continue
                        
                    delta = choices[0].get("delta", {})
                    finish_reason = choices[0].get("finish_reason")
                    
                    if delta and delta.get("reasoning_content"):
                        if not reasoning_started:
                            print("\n🧠 Streaming Reasoning Content:")
                            reasoning_started = True
                            content_started = False
                        reasoning_content += delta["reasoning_content"]
                        print(delta["reasoning_content"], end="", flush=True)
                    
                    if delta and delta.get("content"):
                        if not content_started:
                            print("\n💬 Streaming Response Content:")
                            content_started = True
                            reasoning_started = False
                        final_content += delta["content"]
                        print(delta["content"], end="", flush=True)
                    
                    if finish_reason == "stop":
                        break
                        
            except Exception as chunk_error:
                print(f"\n❌ Error processing chunk: {chunk_error}")
                continue
                
    except Exception as stream_error:
        print(f"\n❌ Error in streaming: {stream_error}")
    
    return reasoning_content, final_content

async def test_agent_api():
    client = AsyncOpenAI(
        base_url="http://localhost:8180/v1",
        api_key="dummy-key"
    )
    
    try:
        # print("🔄 Testing streaming chat completion...")
        
        # response = await client.chat.completions.create(
        #     model="chat_agent",
        #     messages=[
        #         {"role": "user", "content": "Tell me about artificial intelligence"}
        #     ],
        #     stream=True,
        #     extra_body={
        #         "org_id": "test_org_123",
        #         "user_id": "test_user_456", 
        #         "thread_id": "test_thread_789",
        #         "checkpoint_id": "test_checkpoint_101"
        #     }
        # )
        # # async for chunk in response:
        # #     print(chunk.model_dump())
        # await handle_streaming_response(response, "Basic streaming response")
        
        # print("\n\n" + "="*50)
        
        # print("🔄 Testing non-streaming chat completion...")
        
        # response = await client.chat.completions.create(
        #     model="chat_agent",
        #     messages=[
        #         {"role": "user", "content": "What is machine learning?"}
        #     ],
        #     stream=False,
        #     extra_body={
        #         "org_id": "test_org_123",
        #         "user_id": "test_user_456", 
        #         "thread_id": "test_thread_789"
        #     }
        # )
        
        # print("📤 Non-streaming response:")
        # response_dict = response.model_dump()
        # message = response_dict["choices"][0]["message"]
        
        # print(f"Content: {message.get('content', 'No content')}")
        # if message.get('reasoning_content'):
        #     print(f"Reasoning: {message['reasoning_content']}")
        # print(f"Model: {response_dict.get('model', 'Unknown')}")
        # print(f"Finish Reason: {response_dict['choices'][0].get('finish_reason', 'Unknown')}")
        
        # print("\n" + "="*50)
        
        print("🔄 Testing with tools...")

        while True:
            user_input = input("\n\nEnter your query (or 'exit' to quit): ")
            if user_input.lower() == 'exit':
                break
        
            response = await client.chat.completions.create(
                model="chat_agent",
                messages=[
                    {"role": "user", "content": user_input}
                ],
                stream=True,
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "knowledge_search"
                        }
                    }
                ],
                extra_body={
                    "org_id": "test_org",
                    "user_id": "test_user", 
                    "thread_id": "test_thread_789"
                }
            )
            
            await handle_streaming_response(response, "Tool-enabled streaming response")
        
        print("\n\n" + "="*50)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

async def test_models_endpoint():
    client = AsyncOpenAI(
        base_url="http://localhost:8180/v1",
        api_key="dummy-key"
    )
    
    try:
        print("🔄 Testing models endpoint...")
        models = await client.models.list()
        
        print("📤 Available models:")
        for model in models.data:
            print(f"  - {model.id}: {model.name}")
            
    except Exception as e:
        print(f"❌ Error fetching models: {e}")

async def test_error_handling():
    """Test error handling scenarios"""
    client = AsyncOpenAI(
        base_url="http://localhost:8180/v1",
        api_key="dummy-key"
    )
    
    print("🔄 Testing error handling...")
    
    # Test missing model
    try:
        await client.chat.completions.create(
            messages=[{"role": "user", "content": "test"}],
            extra_body={
                "org_id": "test_org",
                "user_id": "test_user", 
                "thread_id": "test_thread"
            }
        )
    except Exception as e:
        print(f"✅ Expected error for missing model: {e}")
    
    # Test invalid model
    try:
        await client.chat.completions.create(
            model="invalid_model",
            messages=[{"role": "user", "content": "test"}],
            extra_body={
                "org_id": "test_org",
                "user_id": "test_user", 
                "thread_id": "test_thread"
            }
        )
    except Exception as e:
        print(f"✅ Expected error for invalid model: {e}")
    
    # Test missing extra_body
    try:
        await client.chat.completions.create(
            model="chat_agent",
            messages=[{"role": "user", "content": "test"}]
        )
    except Exception as e:
        print(f"✅ Expected error for missing extra_body: {e}")
    
    # Test streaming with error
    try:
        response = await client.chat.completions.create(
            model="chat_agent",
            messages=[{"role": "user", "content": "Tell me a very short joke"}],
            stream=True,
            extra_body={
                "org_id": "test_org",
                "user_id": "test_user", 
                "thread_id": "test_thread"
            }
        )
        
        reasoning, content = await handle_streaming_response(response, "Error handling streaming test")
        print(f"\n✅ Streaming completed successfully. Reasoning length: {len(reasoning)}, Content length: {len(content)}")
        
    except Exception as e:
        print(f"❌ Unexpected error in streaming test: {e}")

async def main():
    print("🚀 Starting Agent API Tests\n")
    
    await test_models_endpoint()
    print()
    
    await test_agent_api()
    # print()
    
    # await test_error_handling()
    # print()
    
    print("\n✅ All tests completed!")

if __name__ == "__main__":
    asyncio.run(main())