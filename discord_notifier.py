import requests
import json

WEBHOOK_URL = "https://discord.com/api/webhooks/1459260295439585400/yVZogHEZ2g2u7QRk9xRrtWj7PsYZqyuRlDRWBgyLzwSl_4XJzXdegUlLPXjE5qI5fNZh"

def send_discord(
    title: str,
    description: str,
    color: int = 0x00ff00,
):
    """
    Küld egyszerű embedet Discordra.
    Csak fontos eseményeket.
    """

    payload = {
        "embeds": [
            {
                "title": title,
                "description": description,
                "color": color,
            }
        ]
    }

    headers = {"Content-Type": "application/json"}

    try:
        requests.post(WEBHOOK_URL, data=json.dumps(payload), headers=headers, timeout=5)
    except Exception as e:
        # Nem dobunk itt excet tovább – csak logoljuk
        print(f"[DiscordNotifier] send error: {e}")
