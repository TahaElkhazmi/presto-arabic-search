"""Read-only public-leaderboard fetch (ApiGetLeaderboardRequest).

Separate from the gated submission tool: this endpoint only READS the public
board, so it is safe to run any time. Prints the top teams with scores and
last-submission dates.
"""

import argparse

from kagglesdk import KaggleClient
from kagglesdk.competitions.types.competition_api_service import ApiGetLeaderboardRequest

from common import setup_utf8

COMPETITION = "presto-ai-arabic-e-commerce-search-retrieval-challenge"


def main():
    setup_utf8()
    p = argparse.ArgumentParser()
    p.add_argument("--top", type=int, default=15)
    args = p.parse_args()

    client = KaggleClient()
    api = client.competitions.competition_api_client
    req = ApiGetLeaderboardRequest()
    req.competition_name = COMPETITION
    resp = api.get_leaderboard(req)
    subs = list(resp.submissions)

    print(f"public leaderboard ({len(subs)} teams):\n")
    print(f"{'#':>3}  {'score':>9}  {'last sub (UTC)':<20}  team")
    for i, s in enumerate(subs[: args.top], start=1):
        date = str(s.submission_date)[:19]
        print(f"{i:>3}  {s.score:>9}  {date:<20}  {s.team_name}")


if __name__ == "__main__":
    main()
