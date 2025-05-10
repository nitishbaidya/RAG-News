import feedparser
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from typing import List, Dict, Any, Set
import time
import random
from config import RSS_FEEDS

class NewsArticle:
    def __init__(self, title: str, content: str, url: str, date: datetime, source: str):
        self.title = title
        self.content = content
        self.url = url
        self.date = date
        self.source = source
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "content": self.content,
            "url": self.url,
            "date": self.date.isoformat() if isinstance(self.date, datetime) else self.date,
            "source": self.source
        }

def fetch_rss_feed(feed_url: str, source_name: str) -> List[Dict]:
    """Fetch articles from an RSS feed."""
    feed = feedparser.parse(feed_url)
    entries = []
    
    for entry in feed.entries:
        article = {
            "title": entry.get("title", ""),
            "link": entry.get("link", ""),
            "published": entry.get("published", entry.get("pubDate", "")),
            "source": source_name
        }
        entries.append(article)
    
    return entries

def parse_date(date_str: str) -> datetime:
    """Parse date string into datetime object."""
    try:
        # Try multiple date formats
        formats = [
            "%a, %d %b %Y %H:%M:%S %z",  # RFC 822 format often used in RSS
            "%a, %d %b %Y %H:%M:%S %Z",
            "%Y-%m-%dT%H:%M:%S%z",       # ISO format
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d"
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
                
        # Default to current time if parsing fails
        return datetime.now()
    except Exception:
        return datetime.now()

def fetch_article_content(url: str) -> str:
    """Fetch and extract the main content from an article URL."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "lxml")
        
        # Remove script and style elements
        for script in soup(["script", "style", "header", "footer", "nav"]):
            script.extract()
        
        # Get text content
        text = soup.get_text(separator=" ", strip=True)
        
        # Normalize whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = " ".join(chunk for chunk in chunks if chunk)
        
        return text
    except Exception as e:
        print(f"Error fetching article content from {url}: {e}")
        return ""

def get_all_news_articles(existing_urls: Set[str] = None) -> List[NewsArticle]:
    """
    Fetch and process news articles from all configured RSS feeds.
    
    Args:
        existing_urls: Set of URLs that are already in the database
        
    Returns:
        List of new NewsArticle objects
    """
    all_articles = []
    existing_urls = existing_urls or set()
    
    for source_name, feed_url in RSS_FEEDS.items():
        try:
            print(f"Fetching articles from {source_name}...")
            entries = fetch_rss_feed(feed_url, source_name)
            
            for entry in entries:
                url = entry["link"]
                
                # Skip if this URL already exists in the database
                if url in existing_urls:
                    print(f"Skipping already indexed article: {entry['title']}")
                    continue
                
                # Add a small delay to be polite to the servers
                time.sleep(random.uniform(1, 3))
                
                content = fetch_article_content(url)
                date = parse_date(entry["published"])
                
                if content:
                    article = NewsArticle(
                        title=entry["title"],
                        content=content,
                        url=url,
                        date=date,
                        source=source_name
                    )
                    all_articles.append(article)
                    
        except Exception as e:
            print(f"Error processing feed {source_name}: {e}")
    
    print(f"Fetched {len(all_articles)} new articles")
    return all_articles

if __name__ == "__main__":
    # Test the module
    articles = get_all_news_articles()
    print(f"Fetched {len(articles)} articles")
    for i, article in enumerate(articles[:3]):
        print(f"\nArticle {i+1}:")
        print(f"Title: {article.title}")
        print(f"Source: {article.source}")
        print(f"Date: {article.date}")
        print(f"URL: {article.url}")
        print(f"Content length: {len(article.content)} characters") 