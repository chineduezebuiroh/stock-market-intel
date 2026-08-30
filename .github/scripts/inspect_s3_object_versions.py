from __future__ import annotations

import os
from datetime import datetime, timezone

import boto3
from botocore.exceptions import ClientError


BUCKET = os.environ["S3_BUCKET_DATA"]
PREFIX = os.getenv("S3_PREFIX_DATA", "").strip("/")

KEY = (
    f"{PREFIX}/bars/stocks_daily/CAKE.parquet"
    if PREFIX
    else "bars/stocks_daily/CAKE.parquet"
)

START = datetime(2026, 7, 7, tzinfo=timezone.utc)
END = datetime(2026, 7, 11, tzinfo=timezone.utc)
COMBO_TS = datetime(2026, 7, 9, 3, 0, 53, tzinfo=timezone.utc)

s3 = boto3.client(
    "s3",
    region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
)

print(f"[FORENSICS] bucket: {BUCKET}")
print(f"[FORENSICS] key:    {KEY}")
print(f"[FORENSICS] window: {START.isoformat()} -> {END.isoformat()}")

# Bucket-versioning status is useful but not required.
try:
    status = s3.get_bucket_versioning(Bucket=BUCKET)
    print("\n[FORENSICS] bucket versioning:")
    print(f"  Status:    {status.get('Status', '<not enabled>')}")
    print(f"  MFADelete: {status.get('MFADelete', '<not set>')}")
except ClientError as exc:
    code = exc.response.get("Error", {}).get("Code")
    print(
        "\n[FORENSICS] get_bucket_versioning unavailable "
        f"(permission/error={code}); continuing."
    )

versions = []
delete_markers = []

kwargs = {
    "Bucket": BUCKET,
    "Prefix": KEY,
}

try:
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

except ClientError as exc:
    code = exc.response.get("Error", {}).get("Code")
    msg = exc.response.get("Error", {}).get("Message")

    print(
        "\n[FORENSICS] list_object_versions unavailable."
    )
    print(f"  code={code}")
    print(f"  message={msg}")
    raise SystemExit(2)

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

prior = [
    v for v in versions
    if v["LastModified"] <= COMBO_TS
]

print(
    "\n[FORENSICS] candidate immediately preceding "
    f"{COMBO_TS.isoformat()}:"
)

if prior:
    candidate = max(prior, key=lambda x: x["LastModified"])

    print(
        f"  LastModified={candidate['LastModified'].isoformat()}"
    )
    print(f"  VersionId={candidate['VersionId']}")
    print(f"  Size={candidate['Size']}")
    print(f"  ETag={candidate.get('ETag')}")
else:
    print("  None found.")
