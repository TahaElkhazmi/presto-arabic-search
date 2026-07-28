"""Read-only check of which submission is SELECTED for final scoring.

Lists the competition submissions in the SELECTED group (what Kaggle will use
for the private leaderboard) and, with --all, the full submission history.
Read-only endpoint: it cannot create or change a submission.
"""

import argparse

from kagglesdk import KaggleClient
from kagglesdk.competitions.types.competition_api_service import (
    ApiListSubmissionsRequest,
    SubmissionGroup,
)

from common import setup_utf8

COMPETITION = "presto-ai-arabic-e-commerce-search-retrieval-challenge"


def fetch(api, group):
    req = ApiListSubmissionsRequest()
    req.competition_name = COMPETITION
    req.group = group
    req.page_size = 100
    return list(api.list_submissions(req).submissions)


def show(title, subs):
    print(f"\n=== {title} ({len(subs)}) ===")
    if not subs:
        print("  (none)")
        return
    print(f"  {'date (UTC)':<20}  {'public':>9}  {'private':>9}  {'status':<10}  file")
    for s in subs:
        date = str(s.date)[:19]
        pub = s.public_score if s.public_score else "-"
        priv = s.private_score if s.private_score else "-"
        print(f"  {date:<20}  {str(pub):>9}  {str(priv):>9}  {str(s.status):<10}  {s.file_name}")
        if s.description:
            print(f"      msg: {s.description}")


def main():
    setup_utf8()
    p = argparse.ArgumentParser()
    p.add_argument("--all", action="store_true", help="also list every submission")
    args = p.parse_args()

    api = KaggleClient().competitions.competition_api_client
    show("SELECTED FOR FINAL SCORING", fetch(api, SubmissionGroup.SUBMISSION_GROUP_SELECTED))
    if args.all:
        show("ALL SUBMISSIONS", fetch(api, SubmissionGroup.SUBMISSION_GROUP_ALL))


if __name__ == "__main__":
    main()
