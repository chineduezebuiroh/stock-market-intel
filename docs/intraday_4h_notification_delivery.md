# Intraday 4h notification delivery

Each 4h notification identity has a JSON record at `data/_alerts/delivery_attempts_4h/<job>/<kind>/<sha256>.json`. It stores the identity digest, job, notification type, unique attempt ID, creation and update timestamps, state, and sanitized error category/status. It contains no message, token, or chat ID. Conditional writes prevent a concurrent process or operator from overwriting a changed record. A failed attempt-store operation prevents the POST and produces a warning; core processing continues.

The profile logs the digest, attempt ID and storage revision after recording `INTENT_RECORDED`, then records `POST_STARTED` immediately before calling Telegram. A process can crash in either state without making the POST, or after Telegram receives it; neither state proves delivery. A timeout, connection error or HTTP 5xx is `DELIVERY_UNCONFIRMED`. An HTTP 4xx is `HTTP_REJECTED`, a confirmed HTTP rejection; determine the cause before any further action. All these states suppress automatic repeats of the same identity and appear in warnings. A different identity may be sent. No unsuccessful response advances the confirmed fingerprint.

A successful combo send is marked `CONFIRMED` only after its fingerprint is saved. A fingerprint failure or process crash can leave the attempt unresolved even after delivery. The fingerprint is not modified by reconciliation. Actions summaries and logs contain sanitized identifiers and outcomes. If summary writing fails, the log reports that separately. Older parquet markers from the unmerged draft are not read by this version; there should be no production records from that draft.

## Operator reconciliation

First ensure the originating job is no longer running, inspect the destination and gather evidence. An intent alone does not prove whether a POST started, and `POST_STARTED` does not prove delivery. Never infer nondelivery from a timeout. Configure the same data backend and credentials as the job, then inspect the precise attempt using the full digest from its log:

```sh
python jobs/reconcile_intraday_4h_notification.py --job stocks_intraday_4h --kind repair_telemetry --digest DIGEST
```

Copy the returned `attempt_id` and `revision` exactly. After independently verifying the outcome, run the same command with `--attempt-id ID --revision REVISION --resolution delivered --apply`, or use `--resolution not-delivered` only with evidence of nondelivery. This command does not send a message or alter a confirmed fingerprint. It stores an explicit operator decision; both resolutions continue suppressing the same identity. If a subsequent send is desired, handle it as a separate deliberate operational action, after reviewing duplicate risk. Changed identities continue through normal eligibility rules. An outdated revision or attempt ID, identity mismatch, already reconciled state, or concurrent change is rejected; inspect again before deciding.

The intent and state transitions use conditional writes, but Telegram and storage cannot participate in one transaction. Exactly-once delivery cannot be guaranteed, and an operator decision made while a request is still in flight could be wrong. Do not reconcile active jobs. A successful Telegram response followed by fingerprint or state-store failure still requires investigation. The identity digest for repair telemetry derives from message contents and should be treated as an operational identifier, not as an access-control secret.
