So There files are extracted from a project and we are writing a Indexer API with Fast API solley to make this a mircoservice.

Note : These files just have the flow, but exact methodologies and techniques to do things defer as we have updated lot of methodologies

scope of this microservice is Take any file(PDF, Image, Video, Audio - all in multiple formats) or RAW Text (JSON, HTML,Markdown) or Link and embed that data and store in vector db and possible generate Knowledge graphs. It also is responsible for updating/storing and maintaining the incoming files's extracted content, metadata, source, etc etc in a SQL database (SQLite as of now but will later upgrade to postgres).

Handler classes for qdrant and Neo4j is out of scope of this project. This API Takes the file and gives response with embeddings and knowledge graph data (possibly).

This will be deployed in docker.

SO now start prepping the folder and folder structure and infra for Fast API and this requirements. As of now just prepare the sture and donot inherit antying form the parent project. We will be depricating teh paretn project. So this will be of its own kind. So for now lets build the Fast API essentials for this and start filling those

For the Large Language Model access we have a custom Open AI like API built at /mainpool/Projects/RAG_Application/Application/ModelRouterAPI
