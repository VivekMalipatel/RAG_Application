### Tasks:

1. Modularise the Processing Pipeline
2. Text preprocessing
3. Dynamic Chunk sizing based on Model Context window
4. Update qdrant with processing status 
5. handle document delete events
6. Move parameters for hybrid search, Ollama to env
7. Extend support for multiple model sources
8. Implement proper logging
9. Properly store hugging face models (As of now storing in system cache)
10. Dynamically Adjust Hybrid Search Parameter
11. While Merging Nodes append profiles (Knowledge Graphs)

### Bugs : 

1. System Prompt not properly working (answers are not as per instructions)
2. Decide Proper sizes for chunk, prompt, etc
3. Redis Error on startup
4. Delay for startup
5. 
