#!/usr/bin/env python3
"""Send retrieval prompts through a NIXL prefill server, then a decode server, the way a router
does, and check every answer.

    check.py PREFILL_URL DECODE_URL [--prompts N] [--concurrency C] [--model NAME]

Each prompt buries a random code in 400 to 3000 tokens of filler and asks for it back. The prefill
request carries kv_transfer_params with do_remote_decode and max_tokens=1; the decode request
carries the kv_transfer_params the prefill returned. Exit status is 0 only if every answer holds
its code and every decode request was given transfer parameters.
"""

import argparse
import json
import random
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

FILLER = (
    "The archive holds records of shipments, inspections and repairs, "
    "filed by date and cross-referenced by vessel. "
)


def post(url: str, body: dict) -> dict:
    request = urllib.request.Request(
        f"{url}/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.load(response)


def prompt(index: int) -> tuple[str, str]:
    rng = random.Random(index)
    code = f"{rng.randrange(10**7, 10**8)}"
    before = FILLER * rng.randrange(10, 75)
    after = FILLER * rng.randrange(10, 75)
    text = (
        f"{before}The access code for vault {index} is {code}. Remember it.\n{after}"
        f"Question: what is the access code for vault {index}? Answer with the digits only.\n"
        "Answer:"
    )
    return text, code


def run(index: int, args: argparse.Namespace) -> tuple[bool, str]:
    text, code = prompt(index)
    prefilled = post(
        args.prefill,
        {
            "model": args.model,
            "prompt": text,
            "max_tokens": 1,
            "temperature": 0,
            "kv_transfer_params": {
                "do_remote_decode": True,
                "do_remote_prefill": False,
                "remote_engine_id": None,
                "remote_block_ids": None,
                "remote_host": None,
                "remote_port": None,
            },
        },
    )
    params = prefilled.get("kv_transfer_params")
    if not params:
        return False, f"prompt {index}: prefill returned no kv_transfer_params"
    decoded = post(
        args.decode,
        {
            "model": args.model,
            "prompt": text,
            "max_tokens": 12,
            "temperature": 0,
            "kv_transfer_params": params,
        },
    )
    answer = decoded["choices"][0]["text"]
    if code not in answer:
        return False, f"prompt {index}: wanted {code}, got {answer!r}"
    return True, ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("prefill")
    parser.add_argument("decode")
    parser.add_argument("--prompts", type=int, default=40)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--model", default="m")
    args = parser.parse_args()

    with ThreadPoolExecutor(args.concurrency) as pool:
        results = list(pool.map(lambda i: run(i, args), range(args.prompts)))
    failures = [message for ok, message in results if not ok]
    for message in failures:
        print(message, file=sys.stderr)
    right = len(results) - len(failures)
    print(json.dumps({"prompts": len(results), "right": right}))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
