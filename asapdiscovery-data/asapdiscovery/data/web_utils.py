import abc

import requests
from requests.adapters import HTTPAdapter, Retry


class _BaseWebAPI(abc.ABC):
    """
    A general web API interface base class which sets a requests session and handles API tokens.
    """

    def __init__(self, url, api_version, api_key, **kwargs):
        self.url = url
        self.version = api_version
        self.api_url = f"{self.url}/api/{self.version}"
        self.token = api_key
        self._setup_requests_session()

    def _setup_requests_session(self, retries=3, backoff_factor=0.5):
        self._session = requests.Session()
        self._session.headers.update({self.token_name(): self.token})
        retry = Retry(connect=retries, backoff_factor=backoff_factor)
        adapter = HTTPAdapter(max_retries=retry)
        self._session.mount("https://", adapter)

    @classmethod
    @abc.abstractmethod
    def token_name(cls) -> str:
        """Return the name of the API token which should be used in the header."""
        ...
