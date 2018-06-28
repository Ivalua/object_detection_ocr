import glob
from lxml import etree
import numpy as np
import matplotlib.pyplot as plt
import json

with open('datasets/document.conf') as config_file:
    config = json.load(config_file)

xml_files = glob.glob(config["directory"] + "/*.xml")
num_files = len(xml_files)
print("{} files in dataset".format(num_files))
widths, heights = [], []
ns = {'d': config["namespace"]}
i = 0
for xml_file in xml_files:
    print(xml_file)
    root = etree.parse(xml_file)

    page = root.find(".//d:" + config["page_tag"], ns)
    page_size = [page.get("height"), page.get("width")]

    chars = root.findall(".//d:" + config["char_tag"], ns)
    for c in chars:
        widths.append( int(chars[0].get(config["x2_attribute"])) - int(chars[0].get(config["x1_attribute"])) )
        heights.append( int(chars[0].get(config["y2_attribute"])) - int(chars[0].get(config["y1_attribute"])) )

    i = i + 1

print(np.histogram(np.asarray(widths), bins='auto'))
print(np.histogram(np.asarray(heights), bins='auto'))
plt.hist(np.asarray(widths), bins='auto')
plt.show()
plt.hist(np.asarray(heights), bins='auto')
plt.show()
