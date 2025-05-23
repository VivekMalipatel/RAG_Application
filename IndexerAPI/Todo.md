
### Context

We are starting to build a new API using Fast API This would be a File Indexer Microservice as part of a RAG system.

scope of this microservice is Take any file(PDF, Image, Video, Audio - all in multiple formats) or RAW Text (JSON, HTML,Markdown) or Link and embed that data and store in vector db and possible generate Knowledge graphs. It also is responsible for updating/storing and maintaining the incoming files's extracted content, metadata, source, etc etc in a SQL database (SQLite as of now but will later upgrade to postgres). Handler classes for qdrant and Neo4j is out of scope of this project. This API Takes the file and gives response with embeddings and knowledge graph data (possibly). This will be deployed in docker.

SO now start prepping the folder and folder structure and infra for Fast API and this requirements. The below plan is just initial plan but is still prone to develop a lot

For your context:
/reference_code folder has the code that has been pulled from previous prototype demo, that has the file processing pipeline. But we are totally updgrading that pipeline. 
/pre-tests has the test files that have the testst done on differetn libraries and frameworks on the approach we will be following now. 
The Screenshot attached has the planned flow for this project. 

So you can take the /reference_code with a pinch of salt and /pre-tests are the core concepts that will be scaling and implementing properly with all file type. 

### TASK

Now start building the basics starters for the project and then we will continue with building this microservice. 

### Coding Style and Tech stack

1. It should be completelely modular, Object oriented, well logged
2. No Icons/emoji's in the code or logs
3. As of now we will be sotrng the queue and databse in the same SQlite Db but later upgrading to redis and postgres. So the coding style should be in such a way that with minimal changes we should be able to upgrade.
4. For object storage we will not yet use S3, but use soem temp alternative like how we are using sqlite for postgres aternative but later will be replacing. 

### Plan and Flow

Here is the basic Flow and plan :

📥 Input Sources

The pipeline accepts the following types of inputs:
	•	Document Files:
	•	PDFs, Word Docs, Excel Sheets
	•	CSV, JPEG, MP4, etc.
	•	Raw Text Files
	•	Web Links

⸻

🔄 File Indexer API

All incoming files and links are processed through the File Indexer API. This API is responsible for:
	•	Receiving files or links
	•	Enqueuing them for processing
	•	Storing associated metadata

⸻

📦 Queue and Temporary Storage

After being received by the API:
	•	Files are placed in a processing queue
	•	Raw text or webhook data is stored with metadata in the queue
	•	Files are uploaded to temporary object storage (e.g., S3 or MinIO)

⸻

🧠 Input Type Detection

The Input Type Detector determines the nature of the input and routes it appropriately:
	•	Document Files
	•	Raw Text
	•	Web Links
	•	Audio Files
	•	Video Files

⸻

📄 Processing Document Files
	1.	Convert to Images: Each page or element is transformed into image format.
	2.	Extract Text: Text is extracted from both the original document and the image representation.
	3.	Captioned Images: Images are captioned using vision models.
	4.	Multimodal Embeddings: Both text and image data are passed to a multimodal embedding model.
	5.	Vector Database: Resulting embeddings are stored in a vector DB for retrieval.

⸻

✍️ Processing Raw Text
	1.	Markdown Conversion: Raw text is converted to markdown.
	2.	Extract Text: Directly use the text or enhanced markdown for further analysis.
	3.	LLM Processing:
	•	Ollama LLM is used to analyze or enrich the markdown content.

⸻

🌐 Processing Web Links
	•	Check if YouTube URL:
	•	Yes:
	•	Use Mark it Down → Extract HTML Content → Use Ollama VLM
	•	No:
	•	Browse Page → Extract HTML Content → Use Ollama VLM

⸻

🔊 Processing Audio Files
	•	Decision node: Speech?
	•	If Yes:
	•	Use Mark it Down → Extract Transcription
	•	If No:
	•	Yet to plan

⸻

🎥 Processing Video Files
	•	yet to plan

⸻

🗃 Metadata Schema

Each item processed stores metadata in the following schema:
	1.	Source – The origin of the file or link
	2.	Metadata – Additional descriptive info (e.g., file type, size, tags)
	3.	Indexing Date and Time – Timestamp of processing

### Yet to be plannned

1. How will the embedded data will reach vector db? For this we are yet to write API for vector DB. So this API will be calling the vector db API to store the embeddings
2. Are we planning to build knowlegde graphs? No plan as of now. But will plan once the basic pipeline is done.
3. Audio and Video Processing

### API Design

openapi: 3.0.3
info:
  title: File Indexer API
  version: 1.2.0
  description: API for ingesting files, raw text, and URLs, with support for checking processing status.

servers:
  - url: https://api.example.com/v1

paths:

  /ingest/file:
    post:
      summary: Upload a file for indexing
      requestBody:
        required: true
        content:
          multipart/form-data:
            schema:
              type: object
              properties:
                file:
                  type: string
                  format: binary
                source:
                  type: string
                metadata:
                  type: object
                  additionalProperties: true
      responses:
        '202':
          description: File accepted for processing
          content:
            application/json:
              schema:
                type: object
                properties:
                  id:
                    type: string
                    description: Unique identifier for tracking the submission
                  message:
                    type: string
        '400':
          description: Invalid file format or request

  /ingest/url:
    post:
      summary: Submit a web link for indexing
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                url:
                  type: string
                  format: uri
                space_id:
                  type: string
                source:
                  type: string
                metadata:
                  type: object
                  additionalProperties: true
      responses:
        '202':
          description: URL accepted for processing
          content:
            application/json:
              schema:
                type: object
                properties:
                  id:
                    type: string
                    description: Unique identifier for tracking the submission
                  message:
                    type: string
        '400':
          description: Invalid URL or request format

  /ingest/raw-text:
    post:
      summary: Submit raw text for indexing
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                text:
                  type: string
                  description: The raw text content
                source:
                  type: string
                metadata:
                  type: object
                  additionalProperties: true
      responses:
        '202':
          description: Raw text accepted for processing
          content:
            application/json:
              schema:
                type: object
                properties:
                  id:
                    type: string
                    description: Unique identifier for tracking the submission
                  message:
                    type: string
        '400':
          description: Invalid input

  /status/{id}:
    get:
      summary: Get processing status of a submitted item
      parameters:
        - name: id
          in: path
          required: true
          schema:
            type: string
          description: Unique identifier of the submitted item
      responses:
        '200':
          description: Current status of the item
          content:
            application/json:
              schema:
                type: object
                properties:
                  id:
                    type: string
                  status:
                    type: string
                    enum: [queued, processing, completed, failed]
                  message:
                    type: string
                  result:
                    type: object
                    nullable: true
        '404':
          description: Submission ID not found

components:
  securitySchemes:
    ApiKeyAuth:
      type: apiKey
      in: header
      name: X-API-Key

security:
  - ApiKeyAuth: []


### Current Implementation Plan

1. Lets focus only on Documents with Images and text and focus on Video and audio later. But we will be buidling the temp classes for them too. 
2. 

### How are we doing Emedding and LLM usage?

I have custom built a reverse engineered Open AI like endpoints API that directly works with Openai python library (for text generation and also structured outputs) and for embedding we will desicde if we directly use the openai python library or should cusomtly do the curl commands.