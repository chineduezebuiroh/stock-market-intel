from __future__ import annotations

import os
from datetime import datetime, timezone

import boto3


BUCKET = os.environ["S3_BUCKET_DATA"]
PREFIX = os.getenv("S3_PREFIX_DATA", "").strip("/")

KEY = (
    f"{PREFIX}/bars/stocks_daily/CAKE.parquet"
    if PREFIX
    else "bars/stocks_daily/CAKE.parquet"
)

START = datetime(2026, 7, 7, tzinfo=timezone.utc)
END = datetime(2026, 7, 11, tzinfo=timezone.utc)

s3 = boto3.client(
    "s3",
    region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
)

print(f"[FORENSICS] bucket: {BUCKET}")
print(f"[FORENSICS] key:    {KEY}")
print(f"[FORENSICS] window: {START.isoformat()} -> {END.isoformat()}")

# First check whether versioning is enabled.
status = s3.get_bucket_versioning(Bucket=BUCKET)

print("\n[FORENSICS] bucket versioning:")
print(f"  Status:    {status.get('Status', '<not enabled>')}")
print(f"  MFADelete: {status.get('MFADelete', '<not set>')}")

versions = []
delete_markers = []

kwargs = {
    "Bucket": BUCKET,
    "Prefix": KEY,
}

while True:
    resp = s3.list_object_versions(**kwargs)

    for v in resp.get("Versions", []):
        if v.get("Key") != KEY:
            continue

        lm = v["LastModified"]

        if START <= lm <= END:
            versions.append(v)

    for d in resp.get("DeleteMarkers", []):
        if d.get("Key") != KEY:
            continue

        lm = d["LastModified"]

        if START <= lm <= END:
            delete_markers.append(d)

    if not resp.get("IsTruncated"):
        break

    kwargs["KeyMarker"] = resp.get("NextKeyMarker")
    kwargs["VersionIdMarker"] = resp.get("NextVersionIdMarker")


versions.sort(key=lambda x: x["LastModified"])
delete_markers.sort(key=lambda x: x["LastModified"])

print(f"\n[FORENSICS] versions in window: {len(versions)}")

for v in versions:
    print(
        "  "
        f"{v['LastModified'].isoformat()} "
        f"version_id={v['VersionId']} "
        f"is_latest={v['IsLatest']} "
        f"size={v['Size']} "
        f"etag={v.get('ETag')}"
    )

print(f"\n[FORENSICS] delete markers in window: {len(delete_markers)}")

for d in delete_markers:
    print(
        "  "
        f"{d['LastModified'].isoformat()} "
        f"version_id={d['VersionId']} "
        f"is_latest={d['IsLatest']}"
    )

if not versions:
    print(
        "\n[FORENSICS] No historical object versions were found "
        "for CAKE.parquet in the requested window."
    )
else:
    print(
        "\n[FORENSICS] Candidate version immediately preceding "
        "the 2026-07-09T03:00:53Z combo run:"
    )

    combo_ts = datetime(2026, 7, 9, 3, 0, 53, tzinfo=timezone.utc)

    prior = [
        v for v in versions
        if v["LastModified"] <= combo_ts
    ]

    if prior:
        candidate = max(
            prior,
            key=lambda x: x["LastModified"],
        )

        print(
            f"  LastModified={candidate['LastModified'].isoformat()}"
        )
        print(
            f"  VersionId={candidate['VersionId']}"
        )
        print(
            f"  Size={candidate['Size']}"
        )
        print(
            f"  ETag={candidate.get('ETag')}"
        )
    else:
        print("  None found before combo timestamp.")
