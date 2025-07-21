import os
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET
from tqdm import tqdm

def convert_hazmat_to_voc(source_dir="datasets/hazmatDataset",
                          target_dir="datasets/hazmatVOC"):
    """
    Convert hazmatDataset structure to standard VOCdevkit structure
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)

    # Create VOCdevkit structure
    voc_root = target_path / "VOCdevkit" / "VOC2007"
    voc_root.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (voc_root / "Annotations").mkdir(exist_ok=True)
    (voc_root / "JPEGImages").mkdir(exist_ok=True)
    (voc_root / "ImageSets" / "Main").mkdir(parents=True, exist_ok=True)

    # Process each split
    all_files = []
    split_files = {'train': [], 'val': [], 'test': []}

    for split in ['train', 'valid', 'test']:
        source_split = source_path / split
        if not source_split.exists():
            continue

        # Map split names
        target_split = 'val' if split == 'valid' else split

        # Process images and annotations directly in split folder
        for img_file in tqdm(source_split.glob('*.jpg'), desc=f"Processing {split}"):
            # Get corresponding XML
            xml_name = img_file.stem + '.xml'
            xml_path = source_split / xml_name

            if not xml_path.exists():
                print(f"Warning: No XML for {img_file.name}")
                continue

            # Copy image
            target_img_path = voc_root / "JPEGImages" / img_file.name
            shutil.copy2(img_file, target_img_path)

            # Process and copy XML
            process_xml(xml_path, voc_root / "Annotations" / xml_name,
img_file.name)

            # Add to split list
            file_id = img_file.stem
            split_files[target_split].append(file_id)
            all_files.append(file_id)

    # Create ImageSets files
    for split_name, file_list in split_files.items():
        if file_list:
            split_file = voc_root / "ImageSets" / "Main" /f"{split_name}.txt"
            with open(split_file, 'w') as f:
                f.write('\n'.join(file_list))

    # Create trainval.txt (train + val combined)
    trainval_list = split_files['train'] + split_files['val']
    if trainval_list:
        trainval_file = voc_root / "ImageSets" / "Main" / "trainval.txt"
        with open(trainval_file, 'w') as f:
            f.write('\n'.join(trainval_list))

    print(f"Conversion complete!")
    print(f"Total images: {len(all_files)}")
    print(f"Train: {len(split_files['train'])}")
    print(f"Val: {len(split_files['val'])}")
    print(f"Test: {len(split_files['test'])}")

def process_xml(source_xml, target_xml, image_filename):
    """
    Process XML to ensure VOC compatibility
    """
    tree = ET.parse(source_xml)
    root = tree.getroot()

    # Update filename if needed
    filename_elem = root.find('filename')
    if filename_elem is not None:
        filename_elem.text = image_filename

    # Ensure folder element exists
    folder_elem = root.find('folder')
    if folder_elem is None:
        folder_elem = ET.SubElement(root, 'folder')
        folder_elem.text = 'VOC2007'
    else:
        folder_elem.text = 'VOC2007'

    # Write processed XML
    tree.write(target_xml, encoding='utf-8', xml_declaration=True)

# Run conversion
if __name__ == "__main__":
    convert_hazmat_to_voc()