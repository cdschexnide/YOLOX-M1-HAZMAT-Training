import os
import xml.etree.ElementTree as ET
from pathlib import Path
import json

def analyze_hazmat_dataset():
    dataset_path = Path("datasets/hazmatDataset")

    # Analyze structure
    splits = ['train', 'valid', 'test']
    dataset_info = {}

    for split in splits:
        split_path = dataset_path / split
        if split_path.exists():
            # Look for images directly in the split folder
            image_files = list(split_path.glob('*.jpg')) + list(split_path.glob('*.jpeg'))
            xml_files = list(split_path.glob('*.xml'))

            dataset_info[split] = {
                'images': len(image_files),
                'annotations': len(xml_files),
                'matched': len(image_files) == len(xml_files)
            }

            # Sample XML analysis
            if xml_files:
                sample_xml = xml_files[0]
                tree = ET.parse(sample_xml)
                root = tree.getroot()

                # Extract classes
                classes = set()
                for obj in root.findall('object'):
                    classes.add(obj.find('name').text)

                dataset_info[split]['sample_classes'] = list(classes)

    return dataset_info

# Run analysis
info = analyze_hazmat_dataset()
print(json.dumps(info, indent=2))