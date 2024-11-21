from abc import ABC, abstractmethod
from typing import Dict, List
from collections import defaultdict
from datetime import datetime, timedelta
import time
import logging
import os
import re


class Backend(ABC):
    _api_call_counts: Dict[str, int] = defaultdict(int)
    _last_call_time: Dict[str, Dict[str, datetime]] = {}
    _key_timeout_until: Dict[str, datetime] = defaultdict(lambda: datetime.min)
    _loggers: Dict[str, logging.Logger] = {}

    def __init__(
            self,
            name: str,
            rate_limit: int = 15,
            min_delay: int = 5,
            history_length: int = 8,
            api_key_prefix: str = "",
            verbose: bool = False
    ):
        self.name = name
        self.rate_limit = rate_limit
        self.min_delay = min_delay
        self.history_length = history_length
        self.api_key_prefix = api_key_prefix
        self.verbose = verbose

        # Set up logging only once per backend type
        if name not in Backend._loggers:
            logger = logging.getLogger(f"{name}Backend")
            logger.setLevel(logging.INFO)
            if not logger.handlers:
                ch = logging.StreamHandler()
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                ch.setFormatter(formatter)
                logger.addHandler(ch)
            Backend._loggers[name] = logger

        self.logger = Backend._loggers[name]
        self._initialize_api_keys()

    def handle_rate_limit_error(self, key: str, error_message: str):
        """Handle rate limit error for a specific key."""
        self._set_key_timeout(key, error_message)
        if self.verbose:
            self.logger.info("=== API Key Status After Rate Limit ===")
            for k, status in self.get_key_status().items():
                self.logger.info(f"{k}: {status}")
            self.logger.info("====================================")

    def get_key_status(self) -> Dict[str, Dict]:
        """Get the current status of all API keys."""
        now = datetime.now()
        return {
            f"{self.api_key_prefix}{i+1}": {
                "calls": Backend._api_call_counts[key],
                "last_call": Backend._last_call_time[self.name][key].strftime("%Y-%m-%d %H:%M:%S"),
                "in_timeout": now < Backend._key_timeout_until[key],
                "timeout_remaining": max(0, (Backend._key_timeout_until[key] - now).total_seconds()),
            }
            for i, key in enumerate(self.api_keys)
        }

    def _parse_rate_limit_error(self, error_message: str) -> float:
        """Extract timeout duration from Groq rate limit error message."""
        match = re.search(r'try again in (\d+)m([\d.]+)s', error_message)
        if match:
            minutes, seconds = match.groups()
            return float(minutes) * 60 + float(seconds)
        return 60  # Default timeout if we can't parse the message

    def _set_key_timeout(self, key: str, error_message: str):
        """Set a timeout for a specific API key based on the rate limit error."""
        timeout_duration = self._parse_rate_limit_error(error_message)
        timeout_until = datetime.now() + timedelta(seconds=timeout_duration)
        Backend._key_timeout_until[key] = timeout_until
        self.logger.warning(f"API key {key} in timeout until {timeout_until}")

    def _initialize_api_keys(self):
        """Initialize API keys from environment variables."""
        self.api_keys = []
        i = 1
        while True:
            key = os.environ.get(f"{self.api_key_prefix}{i}")
            if not key:
                break
            self.api_keys.append(key)
            Backend._last_call_time.setdefault(self.name, {})[key] = datetime.min
            i += 1

        if not self.api_keys:
            print(f"No valid API keys found with prefix {self.api_key_prefix}")

        if self.verbose:
            self.logger.info(f"Number of API keys: {len(self.api_keys)}")

    def _respect_rate_limit(self, key: str):
        """Implement rate limiting logic."""
        now = datetime.now()
        time_since_last_call = (now - Backend._last_call_time[self.name][key]).total_seconds()
        calls_in_last_minute = Backend._api_call_counts[key]

        if calls_in_last_minute >= self.rate_limit:
            time_to_wait = 60 - time_since_last_call
            if time_to_wait > 0:
                self.logger.info(f"Rate limiting: Sleeping for {time_to_wait:.2f} seconds")
                time.sleep(time_to_wait)
            Backend._api_call_counts[key] = 0

        if time_since_last_call < self.min_delay:
            sleep_time = self.min_delay - time_since_last_call
            self.logger.info(f"Rate limiting: Sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)

    def _get_next_api_key(self) -> str:
        """Get the next available API key with the lowest usage that's not in timeout."""
        now = datetime.now()
        available_keys = [
            k for k in self.api_keys
            if now >= Backend._key_timeout_until[k]
        ]

        if not available_keys:
            # If all keys are in timeout, wait for the one with the shortest timeout
            next_available_key = min(self.api_keys, key=lambda k: Backend._key_timeout_until[k])
            wait_time = (Backend._key_timeout_until[next_available_key] - now).total_seconds()
            self.logger.info(f"All API keys in timeout. Waiting {wait_time:.2f} seconds for next available key.")
            time.sleep(wait_time)
            return next_available_key

        return min(available_keys, key=lambda k: Backend._api_call_counts[k])

    def _truncate_messages(self, messages: List[Dict]) -> List[Dict]:
        """Keep system message and last N messages."""
        if len(messages) <= self.history_length:
            return messages
        return [messages[0]] + messages[-self.history_length:]

    def _update_api_call_stats(self, key: str):
        """Update API call statistics."""
        Backend._api_call_counts[key] += 1
        Backend._last_call_time[self.name][key] = datetime.now()

        if self.verbose:
            self.logger.info("=== API Key Usage Stats ===")
            for i, k in enumerate(self.api_keys, 1):
                self.logger.info(f"{self.api_key_prefix}{i}: {Backend._api_call_counts[k]} calls")
            self.logger.info("========================")

    @abstractmethod
    def generate(self, messages: List[Dict]) -> str:
        """Generate a response for the given messages."""
        raise NotImplementedError("Subclasses must implement generate()")