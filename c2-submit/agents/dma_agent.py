#!/usr/bin/env python3
"""Exact-optimal policy for the disclosed AEC DMA virtual-cycle model."""

import json
import sys


def main() -> None:
    request = json.load(sys.stdin)

    byte_count = int(request["bytes"])
    concurrency = int(request["concurrency"])
    registered = bool(request.get("registered", False))
    direction = request["direction"]

    # Largest useful legal chunk minimizes ceil(bytes/chunk)-1.
    # On ties, choose the smallest chunk that still gives one chunk.
    if byte_count <= 4096:
        chunk_bytes = 4096
    elif byte_count <= 65536:
        chunk_bytes = 65536
    else:
        chunk_bytes = 1048576

    # The model caps useful parallelism at 2.
    queue_depth = 2 if concurrency >= 2 else 1

    # Registered zero-copy always lowers setup from 100 to 45 and has no
    # countervailing term in the disclosed model.
    use_zero_copy = registered

    # Channel does not affect the disclosed cycle equation. Direction mapping
    # is deterministic and valid.
    channel = 0 if direction == "h2d" else 1

    json.dump(
        {
            "channel": channel,
            "chunk_bytes": chunk_bytes,
            "queue_depth": queue_depth,
            "use_zero_copy": use_zero_copy,
        },
        sys.stdout,
        separators=(",", ":"),
    )


if __name__ == "__main__":
    main()