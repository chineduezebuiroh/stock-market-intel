"""Inspect and explicitly reconcile one 4h delivery attempt; never sends a message."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.notify import _attempt_path_for_digest, _attempt_read, _transition_attempt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', required=True)
    parser.add_argument('--kind', required=True)
    parser.add_argument('--digest', required=True)
    parser.add_argument('--attempt-id', help='Exact attempt ID returned by inspection')
    parser.add_argument('--revision', help='Exact revision returned by inspection')
    parser.add_argument('--resolution', choices=['delivered', 'not-delivered'])
    parser.add_argument('--apply', action='store_true', help='Commit the explicit resolution')
    args = parser.parse_args()
    path = _attempt_path_for_digest(args.job, args.kind, args.digest)
    record, revision = _attempt_read(path)
    if not record:
        parser.error('Attempt not found')
    if (record.get('job'), record.get('notification_type'), record.get('identity_digest')) != (args.job, args.kind, args.digest):
        parser.error('Attempt identity mismatch')
    if not args.apply:
        print(json.dumps({'record': record, 'revision': revision}, sort_keys=True))
        return
    if not args.resolution or not args.attempt_id or not args.revision:
        parser.error('--apply requires --resolution, --attempt-id and --revision')
    if record.get('attempt_id') != args.attempt_id or revision != args.revision:
        parser.error('Attempt ID or revision changed; inspect again')
    if record.get('state') not in {'INTENT_RECORDED', 'POST_STARTED', 'DELIVERY_UNCONFIRMED', 'HTTP_REJECTED'}:
        parser.error('Attempt is not unresolved')
    state = 'OPERATOR_CONFIRMED_DELIVERED' if args.resolution == 'delivered' else 'OPERATOR_ACKNOWLEDGED_NOT_DELIVERED'
    revision = _transition_attempt(path, record, revision, state)
    print(json.dumps({'record': record, 'revision': revision}, sort_keys=True))


if __name__ == '__main__':
    main()
