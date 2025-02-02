import asyncio
from app.core.retrieval.hybrid_retrieval import HybridRetrieval

async def test_multimodal_search():
    retrieval = HybridRetrieval()  # ✅ Removed extra argument

    print("\n🔹 Indexing Multi-Modal Data...")
    retrieval.index_multimodal_data(
        text_data=["AI is transforming healthcare.", "Machine learning in medicine."],
        image_data=["Temp/image/image.png"],
    )

    print("\n🔹 Testing Text-Based Search...")
    text_results = retrieval.search("AI in medicine", mode="text")
    print(f"Results: {text_results}")

    print("\n🔹 Testing Image-Based Search...")
    image_results = retrieval.search("Temp/image/image.png", mode="image")
    print(f"Results: {image_results}")

# Run test
asyncio.run(test_multimodal_search())