Scenario:

First it has Text
Next It has Image
	In the image,
		There is a photo hanging on a wall
				The Photo has cat on the left side
				The Photo has Graph (Image) on the right side
						The graph as some time series representation
						There is Text describing the graph
		There is a Text right side of the wall
		There is Text Under the Photo
Next It has some text Describing the Image

{
    "chunk_id": "chunk-12345",
    "document_id": "doc-98765",
    "user_id": "user-56789",
    "file_name": "complex_document.pdf",
    "file_type": "pdf",
    "chunk_index": 1,
    
    "chunk_text": "Introduction to Advanced AI Concepts.",
    "chunk_image": "base64_encoded_image_data",
    "chunk_audio": null,

    // 🔥 Hybrid Fusion Model (Used for Primary Indexing)
    "chunk_embedding_fusion": [0.89, 0.76, 0.55, 0.31, 0.92],

    // ✅ Source Embeddings (Stored but Not Indexed)
    "chunk_embedding_text": [0.23, 0.67, 0.88, 0.45, 0.12],
    "chunk_embedding_image": [0.78, 0.54, 0.32, 0.90, 0.65],
    "chunk_embedding_audio": null,

    // 🔹 Content Sequence (Spatial-Aware Retrieval)
    "content_sequence": [
        {"type": "text", "index": 1, "text": "Introduction to Advanced AI Concepts.", "relative_position": "before"},
        {"type": "image", "index": 2, "image_ref": "chunk_image"},
        {"type": "text", "index": 3, "text": "This image represents key AI advancements in data analysis.", "relative_position": "after"}
    ],

    // 🔹 Object Map with Explicit References
    "object_map": [
        {
            "label": "photo_frame",
            "bounding_box": [50, 50, 400, 400],
            "objects": [
                {
                    "label": "cat",
                    "bounding_box": [60, 60, 200, 300],
                    "embedding": [0.81, 0.92, 0.45, 0.36, 0.78],
                    "reference": "Described as a cat sitting on the left side."
                },
                {
                    "label": "graph",
                    "bounding_box": [210, 60, 380, 300],
                    "embedding": [0.67, 0.55, 0.89, 0.74, 0.50],
                    "description": "A time-series graph showing AI model accuracy over time.",
                    "extracted_text": "AI model accuracy has improved by 35% in 5 years.",
                    "extracted_text_embedding": [0.76, 0.34, 0.88, 0.65, 0.90],
                    "reference": "Graph located on the right side."
                }
            ]
        },
        {
            "label": "wall_text",
            "bounding_box": [420, 150, 600, 200],
            "extracted_text": "AI-driven analytics for real-world applications.",
            "extracted_text_embedding": [0.54, 0.22, 0.78, 0.33, 0.90],
            "reference": "Text on the right side of the wall."
        },
        {
            "label": "under_photo_text",
            "bounding_box": [50, 420, 400, 450],
            "extracted_text": "AI trends from 2015 to 2025.",
            "extracted_text_embedding": [0.45, 0.63, 0.77, 0.90, 0.55],
            "reference": "Text located under the photo."
        }
    ],

    // 🔹 Layout Hierarchy (Preserves Content Order)
    "layout_hierarchy": [
        {
            "type": "paragraph",
            "text": "Introduction to Advanced AI Concepts.",
            "bounding_box": [50, 20, 600, 80]
        },
        {
            "type": "image",
            "image_ref": "chunk_image",
            "caption": "An AI research poster on a wall.",
            "bounding_box": [50, 100, 600, 500]
        },
        {
            "type": "caption",
            "text": "This image represents key AI advancements in data analysis.",
            "bounding_box": [50, 520, 600, 570]
        }
    ],

    // 🔹 Metadata & Spatial Awareness
    "chunk_metadata": {
        "source_page": 4,
        "ocr_confidence": 0.97,
        "timestamp": 1707132600,
        "created_at": "2025-02-05T14:00:00Z",
        "updated_at": "2025-02-05T15:00:00Z"
    }
}