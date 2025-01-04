from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import time

def fetch_webpage_with_selenium(url):
    # Set up Chrome options
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')

    try:
        driver = webdriver.Chrome(options=chrome_options)
        driver.get(url)
        time.sleep(5)
        html_content = driver.page_source
        driver.quit()
        return html_content
        
    except Exception as e:
        print(f"Error fetching webpage with Selenium: {e}")
        if 'driver' in locals():
            driver.quit()
        return None

def extract_course_urls(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    course_links = soup.find_all('a', class_='link link--large u-font-weight-700')
    urls = [link.get('href') for link in course_links]
    return urls

def save_formatted_urls(urls, filename='course_urls_formatted.txt'):
    base_url = "https://www.gu.se"
    with open(filename, 'w') as f:
        for url in urls:
            formatted_url = f"{base_url}{url}"
            f.write(formatted_url + '\n')

def main():
    urls = [
        "https://www.gu.se/en/study-gothenburg/study-options/find-courses?education_type.keyword=Course&hits=340&subject_area=Art%2C%20Design%2C%20Crafts%20and%20Music",
        "https://www.gu.se/en/study-gothenburg/study-options/find-courses?education_type.keyword=Course&hits=340&subject_area=Business%2C%20Management%20and%20Economics",
        "https://www.gu.se/en/study-gothenburg/study-options/find-courses?education_type.keyword=Course&hits=340&subject_area=Educational%20Sciences",
        "https://www.gu.se/en/study-gothenburg/study-options/find-courses?education_type.keyword=Course&hits=340&subject_area=Health%20Sciences",
        "https://www.gu.se/en/study-gothenburg/study-options/find-courses?education_type.keyword=Course&hits=340&subject_area=Humanities",
        "https://www.gu.se/en/study-gothenburg/study-options/find-courses?education_type.keyword=Course&hits=340&subject_area=Information%20Technology%20and%20Computer%20Science",
        "https://www.gu.se/en/study-gothenburg/study-options/find-courses?education_type.keyword=Course&hits=340&subject_area=Language",
        "https://www.gu.se/en/study-gothenburg/study-options/find-courses?education_level.keyword=Bachelor&education_type.keyword=Course&hits=275&sort=alphabetical&subject_area=Natural%20Sciences%20and%20Mathematics",
        "https://www.gu.se/en/study-gothenburg/study-options/find-courses?education_level.keyword=Master&education_type.keyword=Course&hits=275&sort=alphabetical&subject_area=Natural%20Sciences%20and%20Mathematics",
        "https://www.gu.se/en/study-gothenburg/study-options/find-courses?education_type.keyword=Course&hits=340&subject_area=Social%20Sciences%20and%20Behavioral%20Science"
    ]
    
    all_course_urls = []
    
    for url in urls:
        print(f"\nProcessing URL: {url}")
        html_content = fetch_webpage_with_selenium(url)
        
        if html_content:
            course_urls = extract_course_urls(html_content)
            all_course_urls.extend(course_urls)
            print(f"Found {len(course_urls)} course URLs from this page")
        else:
            print("Failed to fetch webpage content")
    
    # Save all URLs at once
    save_formatted_urls(all_course_urls)
    
    print(f"\nTotal courses found: {len(all_course_urls)}")
    print("\nFirst few formatted URLs:")
    for url in all_course_urls[:5]:
        print(f"https://www.gu.se{url}")

if __name__ == "__main__":
    main()