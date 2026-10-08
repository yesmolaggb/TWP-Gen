from typing import *
import threading

import tavily

from src.config.search_config import search_config
from src.tools._search import SearchClient, SearchResult


_RETRY_KEYWORDS = (
    "429",
    "rate limit",
    "rate_limit",
    "quota",
    "exceeded",
    "exceeds",
    "Too Many Requests",
    "RESOURCE_EXHAUSTED",
    "usage limit",
)


class _TavilyKeyPool:
    """全局 Tavily API key 池，单例模式"""

    _instance: Optional["_TavilyKeyPool"] = None
    _lock = threading.Lock()

    def __init__(self):
        self._keys = list(search_config.tavily_api_keys)
        if not self._keys:
            raise ValueError(
                "No Tavily API keys available. "
                "Please set TAVILY_API_KEYS, TAVILY_API_KEY, TWPGEN_TAVILY_KEY_FILE, or configure tavily_api_key in search.toml."
            )
        self._idx = 0
        self._exhausted: Set[int] = set()

    @classmethod
    def get_instance(cls) -> "_TavilyKeyPool":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def current_key(self) -> str:
        return self._keys[self._idx]

    def rotate(self) -> bool:
        """切换到下一个可用 key，返回是否还有可用的 key"""
        for _ in range(len(self._keys)):
            self._idx = (self._idx + 1) % len(self._keys)
            if self._idx not in self._exhausted:
                return True
        return False

    def mark_exhausted(self, idx: int):
        self._exhausted.add(idx)

    def is_exhausted(self, idx: int) -> bool:
        return idx in self._exhausted

    def all_exhausted(self) -> bool:
        return len(self._exhausted) >= len(self._keys)

    def create_client(self) -> tavily.TavilyClient:
        return tavily.TavilyClient(api_key=self._keys[self._idx])


class TavilySearchClient(SearchClient):
    """Tavily 搜索客户端，通过全局 key 池实现自动切换"""

    def __init__(self):
        self._pool = _TavilyKeyPool.get_instance()

    def search(self, query: str, top_n: int) -> List[SearchResult]:
        if self._pool.all_exhausted():
            print("[Tavily] All API keys exhausted. Skipping search.")
            return []

        max_attempts = len(self._pool._keys) * 2
        for _ in range(max_attempts):
            if self._pool.is_exhausted(self._pool._idx):
                if not self._pool.rotate():
                    break
                continue

            try:
                client = self._pool.create_client()
                search_response = client.search(
                    query=query,
                    max_results=top_n,
                    include_raw_content=True
                )
                results: List[SearchResult] = []
                for item in search_response.get('results', []):
                    results.append(SearchResult(
                        url=item.get('url', ''),
                        title=item.get('title', ''),
                        summary=item.get('content', ''),
                        content=item.get('raw_content', ''),
                        date=''
                    ))
                return results

            except Exception as e:
                err_str = str(e).lower()
                is_rate_limit = any(kw in err_str for kw in _RETRY_KEYWORDS)

                if is_rate_limit:
                    key_hint = self._pool.current_key()[:12]
                    print(f"[Tavily] Rate limit on key {key_hint}..., rotating...")
                    self._pool.mark_exhausted(self._pool._idx)
                    if not self._pool.rotate():
                        print(f"[Tavily] All keys exhausted. Aborting search: {query}")
                        break
                else:
                    key_hint = self._pool.current_key()[:12]
                    print(f"[Tavily] Search error on key {key_hint}...: {e}")
                    break

        return []


if __name__ == "__main__":
    client = TavilySearchClient()
    results = client.search("What is the best pc game in 2024?", 5)
    for result in results:
        print(result.title)
        print(result.summary)
        print(result.content)
        print("---")
