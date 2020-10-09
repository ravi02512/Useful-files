import os
import xml.etree.cElementTree as ET
from PIL import Image
def get_cords(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    xml_list=[]
    classes_names=[]
    for member in root.findall('object'):
          classes_names.append(member[0].text)
          value = (root.find('filename').text,
                   int(root.find('size')[0].text),
                   int(root.find('size')[1].text),
                   member[0].text,
                   int(float(member[4][0].text)),
                   int(float(member[4][1].text)),
                   int(float(member[4][2].text)),
                   int(float(member[4][3].text)))
          xml_list.append(value)
    return xml_list
def coordinateCvt2YOLO(size, box):
        dw = 1. / size[0]
        dh = 1. / size[1]
        # (xmin + xmax / 2)
        x = (box[0] + box[2]) / 2.0
        # (ymin + ymax / 2)
        y = (box[1] + box[3]) / 2.0
        # (xmax - xmin) = w
        w = box[2] - box[0]
        # (ymax - ymin) = h
        h = box[3] - box[1]
        x = x * dw
        w = w * dw
        y = y * dh
        h = h * dh
        return [round(x,3), round(y,3), round(w,3), round(h,3)]
#  list of classes
cls_list=['Tear', 'Shatter', 'Large_tear_or_damage', 'Dislocation', 'Large_dent', 'Scratch_or_spot', 'Dent']

cls_dict=dict(zip(cls_list,list(range(len(cls_list)))))
data=[]
for path in ["annotations/"+i for i in os.listdir("annotations")]:
    xml_list_all=get_cords(path)
    name=os.path.join("labels",os.path.basename(path).replace(".xml",".txt"))
    with open(name,"w") as file:
        for xml_list in xml_list_all:
            ymax=xml_list[-1]
            xmax=xml_list[-2]
            ymin=xml_list[-3]
            xmin=xml_list[-4]
            box=coordinateCvt2YOLO([xml_list[1],xml_list[2]], [xmin,ymin,xmax,ymax])
            file.write(f"{cls_dict[xml_list[3]]} {box[0]} {box[1]} {box[2]} {box[3]}"+"\n")