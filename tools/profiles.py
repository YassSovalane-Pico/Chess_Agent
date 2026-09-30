"""Player-profile tools (chess.com PubAPI, read-only)."""

import requests

from .common import HEADERS, _format_join_date
from .games import _json_object


def get_player_profile(username: str) -> dict:
    """Get the public profile and current ratings of a chess.com player.

    Use this when the user asks about a player, their rating, country,
    join date, followers, or account info.

    NOTE: chess.com has NO official API for a friends list, so a full
    friends list cannot be returned. The closest available data is the
    "followers" count, which this tool returns.

    Args:
        username: The chess.com username (e.g. "hikaru"). Case-insensitive.

    Returns:
        A dict with username, name, title (e.g. "GM", may be None),
        country (country code like "US"), joined (YYYY-MM-DD),
        followers (int or None), and rapid/blitz/bullet
        (current rating or None).
        On failure returns {"error": "human readable message"}.
    """
    username = username.strip()
    if not username:
        return {"error": "Please provide a chess.com username."}

    try:
        player_resp = requests.get(
            f"https://api.chess.com/pub/player/{username}",
            headers=HEADERS,
            timeout=15,
        )
    except requests.RequestException as e:
        return {"error": f"Network error contacting chess.com: {e}"}

    if player_resp.status_code == 404:
        return {"error": f"Player '{username}' not found on chess.com."}
    if player_resp.status_code == 429:
        return {
            "error": "Rate limited by chess.com (429). Please wait a bit and try again."
        }
    if player_resp.status_code != 200:
        return {
            "error": f"chess.com returned HTTP {player_resp.status_code} for profile."
        }

    player = _json_object(player_resp)
    if player is None:
        return {"error": "Could not parse chess.com response."}

    # Country comes as a URL like ".../country/US" -> keep just "US".
    country_url = player.get("country", "")
    country = country_url.rstrip("/").split("/")[-1] if country_url else None

    joined_ts = player.get("joined")
    joined = _format_join_date(joined_ts) if joined_ts else None

    # Ratings live in a second endpoint. If it fails, still return the profile.
    rapid = blitz = bullet = None
    try:
        stats_resp = requests.get(
            f"https://api.chess.com/pub/player/{username}/stats",
            headers=HEADERS,
            timeout=15,
        )
        if stats_resp.status_code == 200:
            stats = _json_object(stats_resp) or {}
            rapid = ((stats.get("chess_rapid") or {}).get("last") or {}).get("rating")
            blitz = ((stats.get("chess_blitz") or {}).get("last") or {}).get("rating")
            bullet = ((stats.get("chess_bullet") or {}).get("last") or {}).get("rating")
    except requests.RequestException:
        pass  # ratings stay None, profile is still useful

    return {
        "username": player.get("username", username),
        "name": player.get("name"),
        "title": player.get("title"),
        "country": country,
        "joined": joined,
        "followers": player.get("followers"),
        "rapid": rapid,
        "blitz": blitz,
        "bullet": bullet,
    }
