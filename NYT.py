# NOTE: THE NEW YORK TIMES
import requests
import pandas as pd
import time
from datetime import datetime

# Define API parameters
api_key = "UPMEUZAACEItC9xAqF7v8dMPxJmPzah3"
search_query = "tesla"
sort = "newest"
start = "20200101"
end = "20250502"
filters = "Business|Technology"

# Initialize an empty list to store all articles
all_articles = []

# Loop through multiple pages
for page in range(1, 301):  # Adjust the range to include more pages (e.g., 1 to 10)
    print(f"Fetching page {page}...")
    url = (f"https://api.nytimes.com/svc/search/v2/articlesearch.json?"
           f"begin_date={start}&end_date={end}&fq={filters}&page={page}&q={search_query}&sort={sort}&api-key={api_key}")

    requestHeaders = {
        "Accept": "application/json"
    }

    response = requests.get(url, headers=requestHeaders)
    data = response.json()
    if response.status_code == 200:
        data = response.json()
        articles = data['response']['docs']
        
        # SANITY CHECK: Ensure 'response' and 'docs' keys exist in the JSON response
        assert 'response' in data, f"Error: 'response' key missing in API response for page {page}."
        assert 'docs' in data['response'], f"Error: 'docs' key missing in API response for page {page}."

        # Append each article to the list
        for article in articles:
            all_articles.append({
                'Date': pd.to_datetime(article['pub_date']).date(),
                'Title': article['headline']['main'],
            })
    else:
        print(f"Error on page {page}: {data['message']}")
        break  # Stop the loop if there's an error
    time.sleep(0)  # Add a delay to avoid hitting the API rate limit

# SANITY CHECK: Ensure at least one article was fetched
assert len(all_articles) > 0, "Error: No articles were fetched from the API."

# Convert the list of articles into a DataFrame
df = pd.DataFrame(all_articles)

# SANITY CHECK: Verify the DataFrame has the expected columns
expected_columns = {'Date', 'Title'}
missing_columns = expected_columns - set(df.columns)
assert not missing_columns, f"Error: Missing expected columns in DataFrame: {missing_columns}"

# Save the DataFrame to a CSV file
df.to_csv('nyt_articles.csv', index=False)