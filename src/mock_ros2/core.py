import time
import queue
from typing import Dict, List, Callable, Any
from threading import Thread, Lock, Event


class Topic:
    def __init__(self, name: str, msg_type: type):
        self.name = name
        self.msg_type = msg_type
        self.subscribers: List[Callable] = []
        self.lock = Lock()
    
    def publish(self, msg: Any):
        with self.lock:
            subscribers = self.subscribers.copy()
        for callback in subscribers:
            try:
                callback(msg)
            except Exception as e:
                print(f"[ERROR] Exception in subscriber callback: {e}")
    
    def subscribe(self, callback: Callable):
        with self.lock:
            self.subscribers.append(callback)


class MockNode:
    def __init__(self, node_name: str):
        self.node_name = node_name
        self.topics: Dict[str, Topic] = {}
        self.topics_lock = Lock()
        self._running = Event()
    
    def create_publisher(self, topic_name: str, msg_type: type) -> Topic:
        if topic_name not in self.topics:
            self.topics[topic_name] = Topic(topic_name, msg_type)
        return self.topics[topic_name]
    
    def create_subscription(self, topic_name: str, msg_type: type, callback: Callable):
        if topic_name not in self.topics:
            self.topics[topic_name] = Topic(topic_name, msg_type)
        self.topics[topic_name].subscribe(callback)
    
    def publish(self, topic_name: str, msg: Any):
        if topic_name in self.topics:
            self.topics[topic_name].publish(msg)
    
    def start(self):
        self._running.set()
    
    def spin(self):
        self._running.set()
        while self._running.is_set():
            time.sleep(0.1)
    
    def stop(self):
        self._running.clear()
    
    def log(self, level: str, message: str):
        print(f"[{level.upper()}] [{self.node_name}] {message}")


class MockSystem:
    def __init__(self):
        self.nodes: List[MockNode] = []
        self.global_topics: Dict[str, Topic] = {}
        self.lock = Lock()
    
    def register_node(self, node: MockNode):
        self.nodes.append(node)
        for topic_name, topic in list(node.topics.items()):
            if topic_name not in self.global_topics:
                self.global_topics[topic_name] = topic
            else:
                # Migrate subscribers to global topic and replace reference
                with topic.lock:
                    subscribers = topic.subscribers.copy()
                for callback in subscribers:
                    self.global_topics[topic_name].subscribe(callback)
                node.topics[topic_name] = self.global_topics[topic_name]
    
    def connect_topics(self, topic_name: str, publisher_node: MockNode, subscriber_node: MockNode, callback: Callable):
        with self.lock:
            if topic_name not in self.global_topics:
                self.global_topics[topic_name] = Topic(topic_name, Any)
        
        with publisher_node.topics_lock:
            publisher_node.topics[topic_name] = self.global_topics[topic_name]
        with subscriber_node.topics_lock:
            subscriber_node.topics[topic_name] = self.global_topics[topic_name]
        self.global_topics[topic_name].subscribe(callback)
    
    def start_all(self):
        for node in self.nodes:
            if hasattr(node, 'start'):
                node.start()
    
    def stop_all(self):
        for node in self.nodes:
            node.stop()
