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
            images_path = split_path / 'images'
            labels_path = split_path / 'labels'
            
            # Count files
            image_files = list(images_path.glob('*.jpg')) + list(images_path.glob('*.jpeg'))
            xml_files = list(labels_path.glob('*.xml'))
            
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
    
    # Read data.yaml for class information
    yaml_path = dataset_path / 'data.yaml'
    if yaml_path.exists():
        with open(yaml_path, 'r') as f:
            yaml_content = f.read()
            dataset_info['yaml_content'] = yaml_content
    
    return dataset_info

# Run analysis
info = analyze_hazmat_dataset()
print(json.dumps(info, indent=2))