import logging
import asyncio
from neo4j import AsyncGraphDatabase
from app.config import settings

class Neo4jSession:
    """Manages a global Neo4j async session with automatic reconnection and health checks."""

    def __init__(self):
        self._driver = None
        self._lock = asyncio.Lock()

    async def connect(self):
        """Initialize Neo4j connection with retries and health checks."""
        async with self._lock:
            if not self._driver:
                max_retries = 5
                for attempt in range(max_retries):
                    try:
                        logging.info("Attempting to connect to Neo4j...")
                        self._driver = AsyncGraphDatabase.driver(
                            uri=settings.NEO4J_URI,
                            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
                        )

                        # Test connection
                        async with self._driver.session() as session:
                            await session.run("RETURN 1")

                        logging.info("Connected to Neo4j successfully.")
                        return
                    except Exception as e:
                        logging.error(f"Neo4j connection failed (Attempt {attempt+1}): {str(e)}")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(2 ** attempt)  # Exponential backoff
                        else:
                            logging.critical("Max retries reached. Neo4j connection failed.")
                            raise

    async def close(self):
        """Close Neo4j session and reset driver."""
        async with self._lock:
            if self._driver:
                await self._driver.close()
                self._driver = None
                logging.info("Neo4j session closed.")

    async def get_session(self):
        """Provides an active Neo4j session with a health check."""
        if not self._driver:
            await self.connect()

        # Health check: Test if the connection is alive
        try:
            async with self._driver.session() as session:
                await session.run("RETURN 1")  # If this fails, reconnect
        except Exception as e:
            logging.warning(f"Neo4j session stale, reconnecting... {str(e)}")
            await self.close()
            await self.connect()

        return self._driver.session()

# Global Neo4j session
neo4j_session = Neo4jSession()