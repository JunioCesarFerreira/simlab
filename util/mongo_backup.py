#!/usr/bin/env python3
"""
MongoDB backup utility — dumps each collection as individual JSON files.

Usage:
    python mongo_backup.py [--uri URI] [--db DB] [--output DIR]

Defaults match the simlab stack (mongodb://localhost:27017/?replicaSet=rs0, db=simlab).
Output structure:
    <output_dir>/<timestamp>/<collection>/<_id>.json
"""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from pymongo import MongoClient


def bson_default(obj):
    if type(obj).__name__ in ("ObjectId", "Decimal128", "Int64", "Int32"):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def backup(uri: str, db_name: str, output_dir: Path) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_root = output_dir / timestamp
    backup_root.mkdir(parents=True, exist_ok=True)

    client = MongoClient(uri)
    try:
        db = client[db_name]
        collections = db.list_collection_names()

        if not collections:
            print(f"No collections found in '{db_name}'.")
            return

        total_docs = 0
        for col_name in sorted(collections):
            col_dir = backup_root / col_name
            col_dir.mkdir(exist_ok=True)

            collection = db[col_name]
            count = 0
            for doc in collection.find():
                doc_id = str(doc.get("_id", count))
                filename = col_dir / f"{doc_id}.json"
                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(doc, f, default=bson_default, ensure_ascii=False, indent=2)
                count += 1

            print(f"  {col_name}: {count} document(s)")
            total_docs += count

        print(f"\nBackup complete → {backup_root}")
        print(f"Collections: {len(collections)}  |  Documents: {total_docs}")
    finally:
        client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backup MongoDB collections to JSON files.")
    parser.add_argument(
        "--uri",
        default=os.getenv("MONGO_URI", "mongodb://localhost:27017/?replicaSet=rs0"),
        help="MongoDB connection URI",
    )
    parser.add_argument(
        "--db",
        default=os.getenv("DB_NAME", "simlab"),
        help="Database name",
    )
    parser.add_argument(
        "--output",
        default="mongo_backup",
        help="Root output directory (default: ./mongo_backup)",
    )
    args = parser.parse_args()

    backup(
        uri=args.uri,
        db_name=args.db,
        output_dir=Path(args.output),
    )
