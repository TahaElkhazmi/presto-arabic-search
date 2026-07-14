"""Submit a run CSV to the Kaggle competition and poll for the leaderboard score.

Auth comes from ~/.kaggle/access_token (or KAGGLE_API_TOKEN env var), the same
mechanism kagglehub used for the data download.
"""

import argparse
import time
from pathlib import Path

import requests
from kagglesdk import KaggleClient
from kagglesdk.competitions.types.competition_api_service import (
    ApiCreateSubmissionRequest,
    ApiListSubmissionsRequest,
    ApiStartSubmissionUploadRequest,
)

from common import setup_utf8

COMPETITION = "presto-ai-arabic-e-commerce-search-retrieval-challenge"


def submit(file_path: Path, message: str) -> None:
    client = KaggleClient()
    api = client.competitions.competition_api_client

    up_req = ApiStartSubmissionUploadRequest()
    up_req.competition_name = COMPETITION
    up_req.content_length = file_path.stat().st_size
    up_req.last_modified_epoch_seconds = int(file_path.stat().st_mtime)
    up_req.file_name = file_path.name
    up = api.start_submission_upload(up_req)
    print(f"upload slot acquired (token …{up.token[-12:] if up.token else '?'})")

    with open(file_path, "rb") as fh:
        resp = requests.put(up.create_url, data=fh, timeout=300)
    resp.raise_for_status()
    print(f"file uploaded ({file_path.stat().st_size:,} bytes, HTTP {resp.status_code})")

    sub_req = ApiCreateSubmissionRequest()
    sub_req.competition_name = COMPETITION
    sub_req.blob_file_tokens = up.token
    sub_req.submission_description = message
    api.create_submission(sub_req)
    print("submission created")


def poll_latest(timeout_s: int = 300) -> None:
    client = KaggleClient()
    api = client.competitions.competition_api_client
    deadline = time.time() + timeout_s
    while True:
        req = ApiListSubmissionsRequest()
        req.competition_name = COMPETITION
        subs = api.list_submissions(req).submissions
        if subs:
            s = subs[0]
            status = str(s.status)
            print(f"latest: file={s.file_name} status={status} "
                  f"public_score={s.public_score!r} desc={s.description!r}")
            if s.public_score or "ERROR" in status.upper() or s.error_description:
                if s.error_description:
                    print(f"error: {s.error_description}")
                return
        if time.time() > deadline:
            print("polling timed out; check the submissions page later")
            return
        time.sleep(15)


def main():
    setup_utf8()
    p = argparse.ArgumentParser()
    p.add_argument("--file", type=Path, required=True)
    p.add_argument("--message", required=True)
    p.add_argument("--poll-only", action="store_true", help="skip submit, just poll")
    args = p.parse_args()
    if not args.poll_only:
        submit(args.file, args.message)
    poll_latest()


if __name__ == "__main__":
    main()
