import os
import re

def clean_script(content):
    """Clean a single script file content."""
    # Find where the actual script content starts (after COLD OPEN)
    start_idx = content.find('COLD OPEN')
    if start_idx == -1:
        # If no COLD OPEN, try to find the first scene heading
        start_idx = re.search(r'\d+\s+(?:INT\.|EXT\.)', content)
        if start_idx:
            start_idx = start_idx.start()
        else:
            return None
    
    # Get content after the start point
    content = content[start_idx:]
    
    # Remove any remaining copyright notices or legal text
    content = re.sub(r'©.*?reserved\.', '', content, flags=re.DOTALL)
    
    # Clean up multiple newlines and spaces
    content = re.sub(r'\n{3,}', '\n\n', content)
    content = re.sub(r' {2,}', ' ', content)
    
    # Remove page numbers and continuation marks
    content = re.sub(r'\n\s*\d+\.\s*\n', '\n', content)
    content = re.sub(r'\(CONTINUED\)|CONTINUED:', '', content)
    
    return content.strip()

def process_scripts(input_dir, output_file):
    """Process all script files and merge them into one file."""
    all_content = []
    
    # Process each file in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.txt'):
            filepath = os.path.join(input_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                
            cleaned_content = clean_script(content)
            if cleaned_content:
                # Add episode separator
                episode_header = f"\n\n{'='*50}\nEPISODE: {filename[:-4]}\n{'='*50}\n\n"
                all_content.append(episode_header + cleaned_content)
    
    # Write all content to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(all_content))

if __name__ == '__main__':
    input_dir = 'sunny/data'
    output_file = 'merged_transcripts.txt'
    process_scripts(input_dir, output_file)
    print(f"Processed scripts have been merged into {output_file}")
