import os
import csv
import shutil
import random
import glob

CANCER_TYPES = [
    '肾癌',
    '前列腺癌',
    '尿路上皮癌'
]

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUTPUT_BASE = os.path.join(PROJECT_ROOT, 'data', 'derived', 'evaluation_dataset')

def read_csv_data(csv_path):
    data = {} # id -> full_row_dict
    # Try different encodings as per project history of encoding issues
    encodings = ['utf-8-sig', 'utf-8', 'gb18030', 'gbk']
    
    for enc in encodings:
        try:
            with open(csv_path, 'r', encoding=enc) as f:
                reader = csv.DictReader(f)
                
                # Normalize headers (strip whitespace)
                if reader.fieldnames:
                    reader.fieldnames = [name.strip() for name in reader.fieldnames]
                
                # Identify columns
                id_col = None
                
                # Look for columns
                possible_headers = reader.fieldnames
                if not possible_headers:
                    continue
                    
                for h in possible_headers:
                    if '门诊号' in h or '住院号' in h:
                        id_col = h
                
                if not id_col:
                    print(f"  Warning: Could not find ID column in {csv_path} with encoding {enc}. Headers: {possible_headers}")
                    continue

                for row in reader:
                    pid = row[id_col].strip()
                    if pid:
                        # Store the full row
                        data[pid] = row
                        
            # If we successfully read some data, return it
            if data:
                return data, reader.fieldnames, id_col
                
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"  Error reading {csv_path} with {enc}: {e}")
            continue
            
    if not data:
        print(f"  Failed to extract data from {csv_path}")
    return {}, [], None

def main():
    print(f"Generating evaluation dataset in {OUTPUT_BASE}...")
    
    if os.path.exists(OUTPUT_BASE):
        print("  Cleaning existing directory...")
        shutil.rmtree(OUTPUT_BASE)
    os.makedirs(OUTPUT_BASE)

    for c_type in CANCER_TYPES:
        print(f"\nProcessing {c_type}...")
        csv_name = f"{c_type}_processed.csv"
        csv_path = os.path.join(PROJECT_ROOT, 'data', 'processed', csv_name)
        img_dir = os.path.join(PROJECT_ROOT, 'outputs', f"{c_type}_processed", 'images')
        
        if not os.path.exists(csv_path):
            print(f"  CSV file not found: {csv_path}")
            continue
            
        # Read Report Data
        reports, fieldnames, id_col_name = read_csv_data(csv_path)
        print(f"  Found {len(reports)} records in CSV.")
        
        if not reports:
            continue
            
        # List Images
        if not os.path.exists(img_dir):
            print(f"  Image directory not found: {img_dir}")
            continue
            
        valid_images = []
        all_files = os.listdir(img_dir)
        
        for img_name in all_files:
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            
            # Extract ID from filename: ID_Hash.png -> ID
            base_name = os.path.splitext(img_name)[0]
            
            # Heuristic: Try to match ID against the dictionary
            candidate_id = base_name.rsplit('_', 1)[0]
            
            if candidate_id in reports:
                valid_images.append((img_name, candidate_id))
            else:
                if base_name in reports:
                    valid_images.append((img_name, base_name))
                else:
                    parts = base_name.split('_')
                    if parts[0] in reports:
                        valid_images.append((img_name, parts[0]))

        print(f"  Found {len(valid_images)} images with matching reports.")
        
        # Sample
        sample_size = 100
        if len(valid_images) == 0:
            print(f"  No images found for {c_type}!")
            continue
            
        if len(valid_images) < sample_size:
            print(f"  Warning: Only found {len(valid_images)} valid images. Taking all.")
            selected = valid_images
        else:
            selected = random.sample(valid_images, sample_size)
            print(f"  Randomly sampled {len(selected)} images.")
            
        # Create Directories
        dest_base = os.path.join(OUTPUT_BASE, c_type)
        images_dest = os.path.join(dest_base, 'images')
        texts_dest = os.path.join(dest_base, 'texts')
        
        os.makedirs(images_dest, exist_ok=True)
        os.makedirs(texts_dest, exist_ok=True)
        
        # Prepare CSV Data
        selected_rows = []
        
        for img_name, pid in selected:
            # Copy Image
            src_img = os.path.join(img_dir, img_name)
            dst_img = os.path.join(images_dest, img_name)
            shutil.copy2(src_img, dst_img)
            
            # Get Data
            row = reports[pid]
            
            # Find conclusion column for text file
            # Assuming '检查结论' is the standard name, but let's check keys
            concl_text = ""
            for k in row.keys():
                if '检查结论' in k:
                    concl_text = row[k]
                    break
            
            # Create Text File
            txt_name = os.path.splitext(img_name)[0] + ".txt"
            dst_txt = os.path.join(texts_dest, txt_name)
            
            with open(dst_txt, 'w', encoding='utf-8') as f:
                f.write(concl_text)
                
            # Add to CSV list, adding the image filename for reference
            row_copy = row.copy()
            row_copy['image_filename'] = img_name
            selected_rows.append(row_copy)
            
        # Write CSV
        csv_dest = os.path.join(dest_base, f"{c_type}_eval_sample.csv")
        # Add 'image_filename' to fieldnames
        output_fieldnames = ['image_filename'] + [f for f in fieldnames if f != 'image_filename']
        
        try:
            with open(csv_dest, 'w', encoding='utf-8-sig', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=output_fieldnames)
                writer.writeheader()
                writer.writerows(selected_rows)
            print(f"  Created CSV at {csv_dest}")
        except Exception as e:
            print(f"  Error writing CSV: {e}")
                
    print("\nDataset generation complete.")

if __name__ == "__main__":
    main()
