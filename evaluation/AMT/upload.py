import base64
import requests
import json
import os
import csv
import re
import random
import pandas as pd

# Replace with your API key
apiKey = 'eab3b6d215fbd570c63cd2011a5d6470'


def uploadImage(filePath):
    with open(filePath, "rb") as file:
        url = "https://api.imgbb.com/1/upload"
        payload = {
            "key": apiKey,
            "image": base64.b64encode(file.read()),
            "name": filePath,
            "expiration": 1728000  # seconds -> 20days
        }
        res = requests.post(url, payload)
    if res.status_code == 200:
        # print("Server Response: " + str(res.status_code))
        # print(json.loads(res.text)["data"]["url"])
        print(filePath + " Image Successfully Uploaded")
        return json.loads(res.text)["data"]["url"]
    else:
        print("ERROR: Server Response: " + str(res.status_code))
        return None


# sense_img_url, input_img1_url, input_img2_url


def write_csv(dir):
    path = "../turk_input.csv"
    os.chdir(dir)
    count = 0
    real_A_url = ''
    real_B_url = ''
    fake_B_url = ''
    with open(path, 'a', newline='') as f:
        csv_write = csv.writer(f)
        # csv_write.writerow(['sense_img_url', 'input_img1_url', 'input_img2_url'])
        for file in os.listdir('.'):
            if re.match(r'.*real_A', file):
                real_A_url = uploadImage(file)
            if re.match(r'.*real_B', file):
                real_B_url = uploadImage(file)
            if re.match(r'.*fake_B', file):
                fake_B_url = uploadImage(file)
            if count == 2:
                count = 0
                if random.random() > 0.5:
                    csv_write.writerow([real_A_url, fake_B_url, real_B_url])
                else:
                    csv_write.writerow([real_A_url, real_B_url, fake_B_url])
            else:
                count += 1
    f.close()


def shuffleCsv():
    path = "turk_input.csv"
    df = pd.read_csv(path, index_col=False)
    shuffled_df = df.sample(frac=1)
    shuffled_df.to_csv("shuffled_turk_input.csv", index=False)


if __name__ == '__main__':
    # write_csv('CycleGAN_googlemap')
    # write_csv('Pix2pix_googlemap')
    # write_csv('BicycleGAN_googlemap')
    shuffleCsv()
