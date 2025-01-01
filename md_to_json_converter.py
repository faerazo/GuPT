import os
import json
import re

def extract_course_code(content):
    # Look for the first occurrence of a course code (e.g., "DIT227")
    match = re.search(r'DIT\d{3}|LT\d{4}|PDA\d{3}|TIA\d{3}', content)
    return match.group(0) if match else None

def convert_md_to_json():
    # Directory paths
    website_dir = 'data/website'
    output_dir = 'data/json'
    output_file = os.path.join(output_dir, 'website.json')
    
    # List to store all course data
    courses = []
    
    # Process each markdown file
    for filename in os.listdir(website_dir):
        if filename.endswith('.md'):
            file_path = os.path.join(website_dir, filename)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # Extract course code
                course_code = extract_course_code(content)
                
                if course_code:
                    course_data = {
                        "course_code": course_code,
                        "website": content
                    }
                    courses.append(course_data)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Write to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(courses, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    convert_md_to_json() 