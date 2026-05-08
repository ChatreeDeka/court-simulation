from __future__ import annotations

from chatree_deka.rag.chroma_store import build_index


def main() -> None:
    count = build_index(force=True)
    print(f"Indexed documents: {count}")


if __name__ == "__main__":
    main()