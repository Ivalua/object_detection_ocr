import cv2
import pytesseract
import json
import glob
from lxml import etree
import time
import lxml.builder

with open('datasets/document.conf') as config_file:
    config = json.load(config_file)

pdf_files = glob.glob(config["directory"] +"/*.jpg")
for i, filename in enumerate(pdf_files):
    start = time.time()
    # read the image and get the dimensions
    img = cv2.imread(filename)
    h, w, _ = img.shape # assumes color image

    # run tesseract, returning the bounding boxes
    boxes = pytesseract.image_to_boxes(img) # also include any config options you use

    root = etree.Element("root", nsmap={None : config["namespace"]})
    p = etree.Element(config["page_tag"], height=str(h), width=str(w))
    root.append( p )


    # draw the bounding boxes on the image
    for b in boxes.splitlines():
        b = b.split(' ')
        # img = cv2.rectangle(img, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)
        c = etree.Element(config["char_tag"])
        c.attrib["x1"] = str(  min(int(b[1]), int(b[3]))  )
        c.attrib["y1"] = str(  min( h - int(b[2]), h - int(b[4]))  )
        c.attrib["x2"] = str(  max(int(b[1]), int(b[3]))  )
        c.attrib["y2"] = str(  max( h - int(b[2]), h - int(b[4]))  )
        c.text = b[0]
        p.append( c )

    print(filename[:-4] + ".xml", time.time() - start)
    etree.ElementTree(root).write(filename[:-4] + ".xml", pretty_print=True, xml_declaration=True, encoding="utf-8")
    # print(etree.tostring(root, pretty_print=True))
    # cv2.imwrite(str(i) + ".jpg", img)
