import httpx
from bs4 import BeautifulSoup
import os

from httpx import TimeoutException
from unidecode import unidecode

headers = {'User-Agent': 'Mozilla/5.0'}
timeout_seconds = 60

image_links_mask = []
image_links_no_mask = []

# Scrap 50 pages
for page in range(100):
    url_mask = f"https://pl.freepik.com/search?demographic=number1&format=search&page={page}&people=include&query=face%20with%20covid%20mask&type=photo#uuid=283a430c-0f19-426e-8fb4-6d093506bb62"
    url_no_mask = f"https://pl.freepik.com/search?demographic=number1&format=search&page={page}&people=include&query=face+person&type=photo#uuid=9e75e487-582a-4ab4-a88e-e31bcad9adab"

    print(f"Page scraping rn: {page} ")

    try:

        response_mask = httpx.get(url_mask, headers=headers, timeout=timeout_seconds)
        response_no_mask = httpx.get(url_no_mask, headers=headers, timeout=timeout_seconds)

        soup_mask = BeautifulSoup(response_mask.text, "html.parser")
        soup_no_mask = BeautifulSoup(response_no_mask.text, "html.parser")

        # Rest of code for scraping images

    except TimeoutException as e:
        print(f"TimeoutException: {e}")
        # Handle the timeout exception here (e.g., retry the request or skip to the next page)


    #Scrap people with mask

    for image_box in soup_mask.select("div.showcase__content.tags-links"):

        img_tag = image_box.select_one("img")

        # Check if the img tag is present and contains a 'data-src' attribute
        if img_tag and 'data-src' in img_tag.attrs:
            title = img_tag.attrs.get("alt", "")
            # Remove punctuation, dots, and replace Polish characters
            title = unidecode(title[:50]).replace(' ', '_')
            title = ''.join(char for char in title if char.isalnum() or char.isspace())

            result = {
                "link": img_tag.attrs["data-src"],
                "title": title
            }

            # Append each image and title to the result array
            image_links_mask.append(result)

    # Scrap people without mask

    for image_box in soup_no_mask.select("div.showcase__content.tags-links"):

        img_tag = image_box.select_one("img")

        # Check if the img tag is present and contains a 'data-src' attribute
        if img_tag and 'data-src' in img_tag.attrs:
            title = img_tag.attrs.get("alt", "")
            # Remove punctuation, dots, and replace Polish characters
            title = unidecode(title[:50]).replace(' ', '_')
            title = ''.join(char for char in title if char.isalnum() or char.isspace())

            result = {
                "link": img_tag.attrs["data-src"],
                "title": title
            }

            # Append each image and title to the result array
            image_links_no_mask.append(result)



print(image_links_mask)
print(image_links_no_mask)

if not os.path.exists('training_dataset'):
    os.makedirs('training_dataset')
    os.makedirs('training_dataset/mask')
    os.makedirs('training_dataset/no_mask')

# Load images with people with mask on

for image_object in image_links_mask:
    # Create a new .png image file
    with open(f"./training_dataset/mask/{image_object['title']}.png", "wb") as file:
        image = httpx.get(image_object["link"])
        # Save the image binary data into the file
        file.write(image.content)
        print(f"Image {image_object['title']} has been scraped")

# Load images with people without mask on

for image_object in image_links_no_mask:
    # Create a new .png image file
    with open(f"./training_dataset/no_mask/{image_object['title']}.png", "wb") as file:
        image = httpx.get(image_object["link"])
        # Save the image binary data into the file
        file.write(image.content)
        print(f"Image {image_object['title']} has been scraped")
