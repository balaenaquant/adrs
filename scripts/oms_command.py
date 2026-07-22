"""Send a command to a running OMS over the aegis NATS control subject and
print its reply.

The OMS subscribes to `oms_command.<oms_id>`, runs the command, and replies
with a JSON status. Uses request-reply so this waits for the outcome.

Usage:
    python scripts/oms_command.py <oms_id>
    python scripts/oms_command.py <oms_id> rebalance --namespace <ns> --timeout 5
"""

import os
import sys
import json
import asyncio
import argparse

from nats_client import NATSClient
from adrs.subjects import oms_command_subject


async def main():
    parser = argparse.ArgumentParser(
        prog="oms_command",
        description="Send a control command to a running OMS instance.",
    )
    parser.add_argument("oms_id")
    parser.add_argument("command", nargs="?", default="rebalance")
    parser.add_argument("--namespace", default=None)
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="seconds to wait for the OMS reply (rebalance does a REST fetch)",
    )
    args = parser.parse_args()

    nats = NATSClient(grpc_addr=os.environ.get("BQ_AEGIS_NATS_GRPC_ADDR"))
    subject = oms_command_subject(args.oms_id, args.namespace)
    body = json.dumps({"command": args.command}).encode()

    try:
        reply = await nats.request(subject, body, timeout=args.timeout)
    except TimeoutError:
        print(
            f"error: no reply from OMS on {subject} within {args.timeout}s",
            file=sys.stderr,
        )
        sys.exit(1)

    result = json.loads(reply.data.decode())
    print(json.dumps(result))
    if result.get("status") != "ok":
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
