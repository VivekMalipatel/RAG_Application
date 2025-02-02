import hashlib
from sqlalchemy import create_engine, Column, String, Integer, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

load_dotenv()

Base = declarative_base()

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class DocumentMetadata(Base):
    """Tracks document versions in PostgreSQL."""
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    document_name = Column(String)
    document_hash = Column(String, unique=True)
    version = Column(Integer, default=1)
    previous_version_id = Column(Integer, ForeignKey("documents.id"), nullable=True)

Base.metadata.create_all(bind=engine)

class MetadataStore:
    """Manages document metadata storage and versioning."""
    
    def __init__(self):
        self.db = SessionLocal()

    def generate_hash(self, content: bytes):
        """Generates a unique SHA-256 hash for the document."""
        return hashlib.sha256(content).hexdigest()

    def add_document(self, user_id: str, doc_name: str, doc_content: bytes):
        """Adds a new document entry to the metadata database."""
        doc_hash = self.generate_hash(doc_content)

        # Check if document already exists
        existing_doc = self.db.query(DocumentMetadata).filter_by(user_id=user_id, document_name=doc_name).first()

        if existing_doc:
            if existing_doc.document_hash == doc_hash:
                return f"Document '{doc_name}' is already stored (no changes detected)."

            # Create a new version of the document
            new_version = DocumentMetadata(
                user_id=user_id,
                document_name=doc_name,
                document_hash=doc_hash,
                version=existing_doc.version + 1,
                previous_version_id=existing_doc.id
            )
            self.db.add(new_version)
            self.db.commit()
            return f"New version {new_version.version} of '{doc_name}' stored."

        # If document is new
        new_doc = DocumentMetadata(
            user_id=user_id,
            document_name=doc_name,
            document_hash=doc_hash,
            version=1
        )
        self.db.add(new_doc)
        self.db.commit()
        return f"Document '{doc_name}' stored successfully."

    def get_latest_version(self, user_id: str, doc_name: str):
        """Retrieves the latest version of a document."""
        doc = self.db.query(DocumentMetadata).filter_by(user_id=user_id, document_name=doc_name).order_by(DocumentMetadata.version.desc()).first()
        return doc.version if doc else None