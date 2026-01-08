import requests

url = "https://www.commonfloor.com/kochi-property/for-rent"
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
}

try:
    print(f"Fetching {url}...")
    response = requests.get(url, headers=headers)
    print(f"Status Code: {response.status_code}")
    
    with open("debug_page.html", "w", encoding="utf-8") as f:
        f.write(response.text)
    
    print("Saved response to debug_page.html")
    
    # Print title
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')
    print(f"Page Title: {soup.title.string if soup.title else 'No Title'}")
    
except Exception as e:
    print(f"Error: {e}")
