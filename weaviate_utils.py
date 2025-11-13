import json
import time
import weaviate
from weaviate.classes.config import Property, DataType

def connect_weaviate(host: str = "http://localhost", port: int = 8090, retries: int = 6, delay: int = 3):
    print(f"Connecting to Weaviate at {host}:{port} ...")
    for attempt in range(retries):
        try:
            client = weaviate.connect_to_local(port=port)

            if client.is_ready():
                print(f"Connected to Weaviate successfully at {host}:{port}")
                return client
            else:
                print(f"Attempt {attempt+1}/{retries}: Weaviate not ready yet.")
        except Exception as e:
            print(f" Attempt {attempt+1}/{retries} failed: {e}")
        time.sleep(delay)
    print(f"Unable to connect to Weaviate after {retries} attempts.")
    return None


def ensure_schema(client, class_name: str = "Document"):
    if client is None:
        print("Weaviate client unavailable — cannot ensure schema.")
        return
    try:
        existing = [c.name for c in client.collections.list_all()]
    except Exception:
        existing = []
    if class_name in existing:
        print(f"Collection '{class_name}' already exists.")
        return
    try:
        client.collections.create(
            name=class_name,
            description="Collection for Folder Relationship Documents",
            properties=[
                Property(name="title", data_type=DataType.TEXT),
                Property(name="content", data_type=DataType.TEXT),
                Property(name="metadata", data_type=DataType.TEXT),
            ],
        )
        print(f"Created new collection schema '{class_name}'.")
    except Exception as e:
        print(f" Failed to ensure schema: {e}")


def upsert_documents(client, docs, class_name: str = "Document"):
    if client is None:
        print(" Skipping Weaviate upsert (no active client).")
        return
    try:
        coll = client.collections.get(class_name)
    except Exception as e:
        print(f"Failed to fetch collection '{class_name}': {e}")
        return

    success_count, fail_count = 0, 0
    for d in docs:
        try:
            props = {
                "title": d.get("title") or "",
                "content": d.get("content") or "",
                "metadata": json.dumps(d.get("metadata", {}), ensure_ascii=False),
            }
            vector = d.get("embedding", None)

            if vector is not None:
                coll.data.insert(properties=props, vector=list(map(float, vector)))
            else:
                coll.data.insert(properties=props)
            success_count += 1
        except Exception as e:
            fail_count += 1
            print(f"Failed to insert document '{d.get('title','unknown')}': {e}")
    print(f"Inserted {success_count} document(s), failed {fail_count}.")


def close_connection(client):
    if client:
        try:
            client.close()
            print("Weaviate connection closed cleanly.")
        except Exception:
            pass


if __name__ == "__main__":
    print("Testing Weaviate connection...")
    c = connect_weaviate()
    if c:
        ensure_schema(c)
        close_connection(c)
    else:
        print("Could not establish a connection.")
