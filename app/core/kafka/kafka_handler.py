import logging
import asyncio
from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
import json

class KafkaHandler:
    """
    Handles Kafka producer and consumer interactions.
    """

    def __init__(self, kafka_bootstrap_servers: str, group_id: str):
        """
        Initializes Kafka producer and consumer.

        Args:
            kafka_bootstrap_servers (str): Kafka server address.
            group_id (str): Consumer group ID.
        """
        self.bootstrap_servers = kafka_bootstrap_servers
        self.group_id = group_id
        self.producer = None
        self.consumer = None

    async def start_producer(self):
        """
        Initializes the Kafka producer.
        """
        self.producer = AIOKafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8")
        )
        await self.producer.start()
        logging.info("Kafka producer started.")

    async def stop_producer(self):
        """
        Stops the Kafka producer.
        """
        if self.producer:
            await self.producer.stop()
            logging.info("Kafka producer stopped.")

    async def start_consumer(self, topic: str):
        """
        Initializes the Kafka consumer with manual commits.

        Args:
            topic (str): Kafka topic to consume.
        """
        self.consumer = AIOKafkaConsumer(
            topic,
            bootstrap_servers=self.bootstrap_servers,
            group_id=self.group_id,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            enable_auto_commit=False  # Disable auto commit for manual processing
        )
        await self.consumer.start()
        logging.info(f"Kafka consumer started for topic: {topic}")

    async def stop_consumer(self):
        """
        Stops the Kafka consumer.
        """
        if self.consumer:
            await self.consumer.stop()
            logging.info("Kafka consumer stopped.")

    async def add_to_queue(self, topic: str, message: dict):
        """
        Produces a message to the Kafka topic.

        Args:
            topic (str): Kafka topic name.
            message (dict): Message to send.
        """
        try:
            if not self.producer:
                await self.start_producer()

            await self.producer.send_and_wait(topic, message)
            logging.info(f"Message sent to topic {topic}: {message}")
        except Exception as e:
            logging.error(f"Error sending message to Kafka: {e}")

    async def consume_message(self, topic: str):
        """
        Consumes a single message from a Kafka topic with manual commit.

        Args:
            topic (str): Kafka topic name.

        Returns:
            dict: The consumed message or None.
        """
        try:
            if not self.consumer:
                await self.start_consumer(topic)

            async for msg in self.consumer:
                try:
                    # Process the message
                    logging.info(f"Consumed message from {topic}: {msg.value}")

                    # Commit offset only after successful processing
                    await self.consumer.commit()
                    logging.info(f"Offset committed for message {msg.offset} in topic {topic}")

                    return msg.value  # Return the message

                except Exception as e:
                    logging.error(f"Error processing message {msg.offset}: {e}")
                    # Don't commit, so it will be retried
                    continue

        except Exception as e:
            logging.error(f"Error consuming message from Kafka: {e}")
            return None