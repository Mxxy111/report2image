import os
import csv
import shutil
import random
import glob

# Configuration
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'derived', 'clinical_utility_dataset_full')

CANCER_TYPES = [
    '肾癌',
    '前列腺癌',
    '尿路上皮癌'
]

TARGET_COUNT = 40

def read_csv_data(csv_path):
    """
    Reads CSV data robustly, handling multiple encodings.
    Returns:
        data: dict {id: row_dict}
        fieldnames: list of strings
        id_col: string (name of the id column)
    """
    data = {} 
    encodings = ['utf-8-sig', 'utf-8', 'gb18030', 'gbk']
    
    for enc in encodings:
        try:
            with open(csv_path, 'r', encoding=enc) as f:
                reader = csv.DictReader(f)
                
                # Normalize headers
                if reader.fieldnames:
                    reader.fieldnames = [name.strip() for name in reader.fieldnames]
                
                # Identify ID column
                id_col = None
                possible_headers = reader.fieldnames
                if not possible_headers:
                    continue
                    
                for h in possible_headers:
                    if '门诊号' in h or '住院号' in h:
                        id_col = h
                        break
                
                if not id_col:
                    print(f"  Warning: Could not find ID column in {csv_path} with encoding {enc}.")
                    continue

                for row in reader:
                    pid = row[id_col].strip()
                    if pid:
                        data[pid] = row
                        
            if data:
                return data, reader.fieldnames, id_col
                
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"  Error reading {csv_path} with {enc}: {e}")
            continue
            
    print(f"  Failed to extract data from {csv_path}")
    return {}, [], None

def calculate_complexity_score(row):
    """
    Calculate a complexity score based on report length and keywords.
    """
    conclusion = row.get('检查结论', '') or ''
    findings = row.get('检查所见', '') or ''
    
    if not conclusion and not findings:
        return 0.0

    score = len(conclusion) * 1.0  # Base score from length
    
    # Keywords boosting score
    keywords = [
        '转移', '多发', '多处', '广泛', '全身', 
        '骨', '淋巴结', '肺', '肝',  # Common metastasis sites
        '增高', '代谢', '异常'
    ]
    
    for kw in keywords:
        count = conclusion.count(kw)
        score += count * 20  # Significant boost for key terms in conclusion
        score += findings.count(kw) * 5 # Smaller boost for findings

    return score

def main():
    print(f"Selecting complex cases from FULL dataset into {OUTPUT_DIR}...")
    
    if os.path.exists(OUTPUT_DIR):
        print("  Cleaning existing output directory...")
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)
    
    all_selected_rows = []

    for c_type in CANCER_TYPES:
        print(f"\nProcessing {c_type}...")
        
        # Source paths (Raw Data)
        csv_name = f"{c_type}_processed.csv"
        csv_path = os.path.join(PROJECT_ROOT, 'data', 'processed', csv_name)
        img_dir = os.path.join(PROJECT_ROOT, 'outputs', f"{c_type}_processed", 'images')
        
        if not os.path.exists(csv_path):
            print(f"  CSV file not found: {csv_path}")
            continue
        
        if not os.path.exists(img_dir):
            print(f"  Image directory not found: {img_dir}")
            continue

        # 1. Load CSV Data
        reports, fieldnames, id_col = read_csv_data(csv_path)
        print(f"  Found {len(reports)} records in CSV.")
        
        if not reports:
            continue

        # 2. Match Images to Reports
        valid_matches = [] # list of (image_filename, row_dict)
        all_files = os.listdir(img_dir)
        
        matched_count = 0
        for img_name in all_files:
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            
            # Extract ID from filename
            base_name = os.path.splitext(img_name)[0]
            
            # Heuristics to find ID
            # 1. ID_Hash
            candidate_id = base_name.rsplit('_', 1)[0]
            
            row = None
            if candidate_id in reports:
                row = reports[candidate_id]
            elif base_name in reports:
                row = reports[base_name]
            else:
                parts = base_name.split('_')
                if parts[0] in reports:
                    row = reports[parts[0]]
            
            if row:
                # Add image filename to row for convenience
                # We need to copy row to avoid mutating shared dict if multiple images map to same ID (unlikely but possible)
                row_copy = row.copy()
                row_copy['image_filename'] = img_name
                valid_matches.append(row_copy)
                matched_count += 1

        print(f"  Matched {matched_count} images to reports.")

        # 3. Score and Sort
        for row in valid_matches:
            row['complexity_score'] = calculate_complexity_score(row)
            
        # Sort descending by score
        valid_matches.sort(key=lambda x: x['complexity_score'], reverse=True)
        
        # 4. Select Top N
        selected = valid_matches[:TARGET_COUNT]
        if not selected:
             print(f"  No valid matches found for {c_type}")
             continue

        print(f"  Selected top {len(selected)} complex cases (Score range: {selected[0]['complexity_score']:.1f} - {selected[-1]['complexity_score']:.1f}).")

        # 5. Output
        dest_cancer_dir = os.path.join(OUTPUT_DIR, c_type)
        dest_images = os.path.join(dest_cancer_dir, 'images')
        dest_texts = os.path.join(dest_cancer_dir, 'texts')
        os.makedirs(dest_images, exist_ok=True)
        os.makedirs(dest_texts, exist_ok=True)
        
        selected_for_csv = []

        for row in selected:
            img_name = row['image_filename']
            
            # Copy Image
            src_img = os.path.join(img_dir, img_name)
            dst_img = os.path.join(dest_images, img_name)
            shutil.copy2(src_img, dst_img)
            
            # Create Text File
            # Extract conclusion
            concl_text = ""
            for k in row.keys():
                if '检查结论' in k:
                    concl_text = row[k]
                    break
            
            txt_name = os.path.splitext(img_name)[0] + ".txt"
            dst_txt = os.path.join(dest_texts, txt_name)
            with open(dst_txt, 'w', encoding='utf-8') as f:
                f.write(concl_text)
                
            # Prepare for Metadata CSV
            row['cancer_type'] = c_type
            selected_for_csv.append(row)
            all_selected_rows.append(row)

        # Generate CSV for this cancer type
        type_csv_path = os.path.join(dest_cancer_dir, f"{c_type}_complex_full.csv")
        # Ensure image_filename is in headers
        out_fields = ['image_filename'] + [f for f in fieldnames if f != 'image_filename']
        if 'complexity_score' not in out_fields: out_fields.append('complexity_score') # Add score to output
        
        try:
            with open(type_csv_path, 'w', encoding='utf-8-sig', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=out_fields, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(selected_for_csv)
        except Exception as e:
            print(f"  Error writing CSV for {c_type}: {e}")

    # Generate Merged Metadata CSV
    if all_selected_rows:
        print("\nGenerating consolidated metadata CSV...")
        meta_csv_path = os.path.join(OUTPUT_DIR, 'clinical_utility_metadata_full.csv')
        
        # Determine consolidated headers
        # Use fieldnames from last iteration (should be roughly same) or union
        all_keys = set()
        for r in all_selected_rows:
            all_keys.update(r.keys())
        
        # Order them nicely
        ordered_keys = ['cancer_type', 'image_filename', 'complexity_score', '门诊号/住院号', '姓名', '诊断', '检查结论', '检查所见']
        # Add remaining keys
        for k in all_keys:
            if k not in ordered_keys:
                ordered_keys.append(k)
        
        with open(meta_csv_path, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=ordered_keys, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(all_selected_rows)
            
        print(f"  Saved to {meta_csv_path}")

    print("\nFull Dataset Selection Complete.")

if __name__ == "__main__":
    main()
