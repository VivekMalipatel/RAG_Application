import logging
from neo4j import AsyncGraphDatabase
from app.config import settings
import asyncio

class Neo4jSession:
    """
    Manages a global Neo4j async session.
    """

    def __init__(self):
        self._driver = None
        self._lock = asyncio.Lock()

    async def connect(self):
        """Initialize Neo4j driver."""
        async with self._lock:
            if not self._driver:
                try:
                    self._driver = AsyncGraphDatabase.driver(
                        settings.NEO4J_URI,
                        auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
                    )
                    # Test connection
                    async with self._driver.session() as session:
                        await session.run("RETURN 1")
                    logging.info("Connected to Neo4j successfully.")
                except Exception as e:
                    logging.error(f"Failed to initialize Neo4j session: {str(e)}")
                    raise

    async def close(self):
        """Close Neo4j session."""
        async with self._lock:
            if self._driver:
                await self._driver.close()
                self._driver = None
                logging.info("Neo4j session closed.")

    async def get_session(self):
        """Provides an active session for queries."""
        if not self._driver:
            await self.connect()
        return self._driver.session()

# Instantiate a global Neo4j session
neo4j_session = Neo4jSession()