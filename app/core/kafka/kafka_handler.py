import logging
from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
import json

class KafkaHandler:
    """
    Handles Kafka message production and consumption.
    """

    def __init__(self, kafka_server):
        """
        Initializes Kafka producer.

        Args:
            kafka_server (str): Kafka broker address.
        """
        self.kafka_server = kafka_server
        self.producer = AIOKafkaProducer(bootstrap_servers=self.kafka_server)

    async def start(self):
        """Starts the Kafka producer."""
        await self.producer.start()

    async def stop(self):
        """Stops the Kafka producer."""
        await self.producer.stop()

    async def publish(self, topic: str, message: dict):
        """
        Publishes a message to a Kafka topic.

        Args:
            topic (str): Kafka topic name.
            message (dict): Message payload.
        """
        try:
            await self.start()
            await self.producer.send_and_wait(topic, json.dumps(message).encode("utf-8"))
            await self.stop()
            logging.info(f"Message published to Kafka topic '{topic}': {message}")
        except Exception as e:
            logging.error(f"Error publishing to Kafka topic '{topic}': {e}")