#!/usr/bin/env python3

import os
import xml.etree.ElementTree as ET
from collections import Counter

def find_all_classes():
    annotations_dir = "datasets/hazmatVOC/VOCdevkit/VOC2007/Annotations"
    class_names = []
    
    for xml_file in os.listdir(annotations_dir):
        if xml_file.endswith('.xml'):
            xml_path = os.path.join(annotations_dir, xml_file)
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                
                for obj in root.findall('object'):
                    name = obj.find('name').text.strip()
                    class_names.append(name)
            except Exception as e:
                print(f"Error reading {xml_file}: {e}")
    
    # Count occurrences
    class_counts = Counter(class_names)
    
    print("All classes found in annotations:")
    for class_name, count in class_counts.most_common():
        print(f"  {class_name}: {count} instances")
    
    print(f"\nTotal unique classes: {len(class_counts)}")
    return list(class_counts.keys())

if __name__ == "__main__":
    classes = find_all_classes()