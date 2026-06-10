"""
Member A — Step 1: Fetch articles from RSS feeds and NewsAPI
"""

import feedparser
from newsapi import NewsApiClient
from newspaper import Article
from dotenv import load_dotenv
import os

load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# RSS feeds to pull from
RSS_FEEDS = [
    "https://techcrunch.com/feed/",
    "https://www.wired.com/feed/rss",
    "https://www.theverge.com/rss/index.xml",
    "https://feeds.arstechnica.com/arstechnica/index",
    "https://venturebeat.com/feed/",
]


def fetch_from_rss(feeds: list[str]) -> list[dict]:
    """Pull articles from RSS feeds."""
    articles = []
    for url in feeds:
        feed = feedparser.parse(url)
        for entry in feed.entries:
            articles.append({
                "title": entry.get("title", ""),
                "url": entry.get("link", ""),
                "source": feed.feed.get("title", "Unknown"),
                "date": entry.get("published", ""),
                "content": "",  # filled by fetch_full_content
            })
    print(f"[Fetcher] RSS: fetched {len(articles)} articles")
    return articles


def fetch_from_newsapi(query: str = "technology", page_size: int = 20) -> list[dict]:
    """Pull top headlines from NewsAPI."""
    if not NEWS_API_KEY:
        print("[Fetcher] NEWS_API_KEY not set — skipping NewsAPI")
        return []

    client = NewsApiClient(api_key=NEWS_API_KEY)
    response = client.get_everything(q=query, language="en", page_size=page_size)
    articles = []
    for item in response.get("articles", []):
        articles.append({
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "source": item.get("source", {}).get("name", "Unknown"),
            "date": item.get("publishedAt", ""),
            "content": item.get("content", ""),
        })
    print(f"[Fetcher] NewsAPI: fetched {len(articles)} articles")
    return articles


def fetch_full_content(article: dict) -> dict:
    """Use newspaper3k to extract full article text."""
    try:
        a = Article(article["url"])
        a.download()
        a.parse()
        article["content"] = a.text
    except Exception as e:
        print(f"[Fetcher] Failed to fetch {article['url']}: {e}")
    return article


def fetch_all_articles() -> list[dict]:
    """Main entry point — fetch from all sources."""
    articles = (
        fetch_from_rss(RSS_FEEDS)
        + fetch_from_newsapi(query="OpenAI", page_size=20)
        + fetch_from_newsapi(query="artificial intelligence", page_size=20)
        + fetch_from_newsapi(query="technology startups", page_size=10)
    )
    articles = [fetch_full_content(a) for a in articles if a["url"]]
    return articles


if __name__ == "__main__":
    results = fetch_all_articles()
    print(f"Total articles fetched: {len(results)}")
