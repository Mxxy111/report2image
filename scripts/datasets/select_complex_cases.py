import os
import csv
import shutil
import glob

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
SOURCE_DIR = os.path.join(PROJECT_ROOT, 'data', 'derived', 'evaluation_dataset')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'derived', 'clinical_utility_dataset')

CANCER_TYPES = [
    '肾癌',
    '前列腺癌',
    '尿路上皮癌'
]

TARGET_COUNT = 40

def calculate_complexity_score(row):
    """
    Calculate a complexity score based on report length and keywords.
    Heuristic:
    1. Length of conclusion (longer = more findings)
    2. Keywords indicating metastasis or multiple lesions
    """
    conclusion = row.get('检查结论', '') or ''
    findings = row.get('检查所见', '') or ''
    full_text = conclusion + findings
    
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
        
        # Smaller boost for findings (as findings are verbose)
        score += findings.count(kw) * 5

    return score

def main():
    print(f"Selecting complex cases from {SOURCE_DIR}...")
    
    if os.path.exists(OUTPUT_DIR):
        print("  Cleaning existing output directory...")
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)
    
    all_selected_rows = []

    for c_type in CANCER_TYPES:
        print(f"\nProcessing {c_type}...")
        
        # Source paths
        src_cancer_dir = os.path.join(SOURCE_DIR, c_type)
        csv_path = os.path.join(src_cancer_dir, f"{c_type}_eval_sample.csv")
        images_dir = os.path.join(src_cancer_dir, 'images')
        texts_dir = os.path.join(src_cancer_dir, 'texts')
        
        if not os.path.exists(csv_path):
            print(f"  CSV not found: {csv_path}")
            continue
            
        # Read CSV
        rows = []
        try:
            with open(csv_path, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
        except Exception as e:
            print(f"  Error reading CSV: {e}")
            continue
            
        print(f"  Loaded {len(rows)} cases.")
        
        # Score and Sort
        for row in rows:
            row['complexity_score'] = calculate_complexity_score(row)
            
        # Sort descending by score
        rows.sort(key=lambda x: x['complexity_score'], reverse=True)
        
        # Select Top N
        selected = rows[:TARGET_COUNT]
        print(f"  Selected top {len(selected)} complex cases (Score range: {selected[0]['complexity_score']:.1f} - {selected[-1]['complexity_score']:.1f}).")
        
        # Output paths
        dest_cancer_dir = os.path.join(OUTPUT_DIR, c_type)
        dest_images = os.path.join(dest_cancer_dir, 'images')
        dest_texts = os.path.join(dest_cancer_dir, 'texts')
        os.makedirs(dest_images, exist_ok=True)
        os.makedirs(dest_texts, exist_ok=True)
        
        for row in selected:
            img_name = row.get('image_filename')
            if not img_name:
                continue
                
            # Copy Image
            src_img = os.path.join(images_dir, img_name)
            dst_img = os.path.join(dest_images, img_name)
            if os.path.exists(src_img):
                shutil.copy2(src_img, dst_img)
            else:
                print(f"    Warning: Image missing {src_img}")
                
            # Copy Text
            txt_name = os.path.splitext(img_name)[0] + ".txt"
            src_txt = os.path.join(texts_dir, txt_name)
            dst_txt = os.path.join(dest_texts, txt_name)
            if os.path.exists(src_txt):
                shutil.copy2(src_txt, dst_txt)
            
            # Add type for merged CSV
            row['cancer_type'] = c_type
            all_selected_rows.append(row)

    # Generate Merged Metadata CSV
    if all_selected_rows:
        print("\nGenerating consolidated metadata CSV...")
        meta_csv_path = os.path.join(OUTPUT_DIR, 'clinical_utility_metadata.csv')
        
        # Get all fields + extra ones
        fieldnames = list(all_selected_rows[0].keys())
        # Ensure 'cancer_type' and 'complexity_score' are included and maybe moved to front/back
        if 'cancer_type' not in fieldnames: fieldnames.append('cancer_type')
        if 'complexity_score' not in fieldnames: fieldnames.append('complexity_score')
        
        # Reorder slightly for readability
        priority_fields = ['cancer_type', 'image_filename', 'complexity_score']
        for f in reversed(priority_fields):
            if f in fieldnames:
                fieldnames.remove(f)
                fieldnames.insert(0, f)
        
        with open(meta_csv_path, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_selected_rows)
            
        print(f"  Saved to {meta_csv_path}")

    print("\nClinical Utility Dataset selection complete.")

if __name__ == "__main__":
    main()
