import requests, json

url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries"
resp = requests.get(url, timeout=10)
print(resp.status_code)
print(json.dumps(resp.json(), indent=2)[:3000])