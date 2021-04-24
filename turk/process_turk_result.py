import re
import pandas as pd


def model_classify(image_url):
    if re.match(r'.*Bicycle', image_url):
        return 'BicycleGAN'
    if re.match(r'.*Pix2pix', image_url):
        return 'Pix2pix'
    if re.match(r'.*Cycle', image_url):
        return 'CycleGAN'


def label_classify(image_url):
    if re.match(r'.*fake', image_url):
        return False
    if re.match(r'.*real', image_url):
        return True


if __name__ == '__main__':
    path = "sample_batch_results.csv"
    df = pd.read_csv(path, index_col=False)
    right_labels = {'BicycleGAN': 0, 'Pix2pix': 0, 'CycleGAN': 0}
    wrong_labels = {'BicycleGAN': 0, 'Pix2pix': 0, 'CycleGAN': 0}
    wasted = 0
    for index, row in df.iterrows():
        # real_A = row["Input.sense_img_url"]
        # the label of the image that workers picked
        selected_label = row['Answer.category.label']
        if selected_label == 1:
            img = row["Input.input_img1_url"]
        elif selected_label == 2:
            img = row["Input.input_img2_url"]
        else:
            wasted += 1
            continue
        label = label_classify(img)
        model = model_classify(img)
        if not label:
            wrong_labels[model] += 1
        else:
            right_labels[model] += 1
    # more wrongly classified labels suggest better performance
    print("Rightly classified: ", right_labels)
    print("wrongly classified: ", wrong_labels)
