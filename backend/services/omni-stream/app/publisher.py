"""RabbitMQ event publisher for live detections."""

import json
import logging
import os
import threading

import pika

log = logging.getLogger("omni-stream.publisher")


class EventPublisher:
    def __init__(self):
        self._host = os.getenv("RABBITMQ_HOST", "rabbitmq")
        self._port = int(os.getenv("RABBITMQ_PORT", "5672"))
        self._user = os.getenv("RABBITMQ_USER", "omni")
        self._password = os.getenv("RABBITMQ_PASS", "omni_secret")
        self._vhost = os.getenv("RABBITMQ_VHOST", "/")
        self._exchange = os.getenv("RABBITMQ_EXCHANGE", "omni.events")
        self._connection = None
        self._channel = None
        self._lock = threading.Lock()
        self._connect()

    def _connect(self):
        try:
            creds = pika.PlainCredentials(self._user, self._password)
            params = pika.ConnectionParameters(
                host=self._host,
                port=self._port,
                virtual_host=self._vhost,
                credentials=creds,
                heartbeat=30,
                blocked_connection_timeout=10,
            )
            self._connection = pika.BlockingConnection(params)
            self._channel = self._connection.channel()
            self._channel.exchange_declare(exchange=self._exchange, exchange_type="fanout", durable=True)
            log.info("RabbitMQ connected: %s:%d", self._host, self._port)
        except Exception as exc:
            log.warning("RabbitMQ connection failed: %s - events will be dropped", exc)
            self._channel = None

    def publish(self, event: dict):
        with self._lock:
            if not self._channel:
                self._connect()
                if not self._channel:
                    return
            try:
                self._channel.basic_publish(
                    exchange=self._exchange,
                    routing_key="",
                    body=json.dumps(event),
                    properties=pika.BasicProperties(content_type="application/json", delivery_mode=1),
                )
            except Exception as exc:
                log.warning("Publish failed: %s - reconnecting", exc)
                self._channel = None
                self._connect()
