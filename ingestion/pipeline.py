"""
Member A — Main ingestion pipeline (orchestrates all steps)
Runs on a schedule every 6-12 hours via APScheduler
"""

from fetcher import fetch_all_articles
from cleaner import clean_articles
from chunker import chunk_all_articles
from embedder import embed_and_store
from apscheduler.schedulers.blocking import BlockingScheduler
import time


def run_ingestion():
    """Full pipeline: fetch → clean → chunk → embed → store."""
    print(f"\n{'='*50}")
    print(f"[Pipeline] Starting ingestion run...")
    print(f"{'='*50}")

    # Step 1: Fetch
    articles = fetch_all_articles()
    if not articles:
        print("[Pipeline] No articles fetched. Skipping.")
        return

    # Step 2: Clean
    articles = clean_articles(articles)

    # Step 3: Chunk (hierarchical — parents for context, children for retrieval)
    parents, children = chunk_all_articles(articles)

    # Step 4 & 5: Embed + Store
    embed_and_store(parents, children)

    print(f"[Pipeline] Done. {len(parents)} parent + {len(children)} child chunks in ChromaDB.\n")


def run_scheduled(interval_hours: int = 6):
    """Run ingestion on a schedule."""
    scheduler = BlockingScheduler()
    scheduler.add_job(run_ingestion, "interval", hours=interval_hours)
    print(f"[Pipeline] Scheduler started — running every {interval_hours} hours")
    run_ingestion()  # run once immediately
    scheduler.start()


if __name__ == "__main__":
    import sys
    if "--schedule" in sys.argv:
        run_scheduled(interval_hours=6)
    else:
        run_ingestion()
