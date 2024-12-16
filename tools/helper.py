import os
import base64
from weasyprint import HTML, CSS
import os
import markdown2
from datetime import datetime
import colorsys

def MoveCharts(src_dir="charts", dest_dir="Dumped Charts"):
    """
    Moves files from the 'charts' directory to the 'Dumped Charts' directory,
    renaming files to avoid overwriting existing files in the destination directory.
    """
    import os
    import shutil
    # Ensure the source directory exists
    if not os.path.exists(src_dir):
        print(f"Source directory '{src_dir}' does not exist.")
        return

    # Create the destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)

    # Scan the source directory for files
    for file_name in os.listdir(src_dir):
        src_file_path = os.path.join(src_dir, file_name)

        # Skip directories and handle only files
        if os.path.isfile(src_file_path):
            dest_file_path = os.path.join(dest_dir, file_name)

            # If a file with the same name exists in the destination, rename it
            base_name, ext = os.path.splitext(file_name)
            counter = 1
            while os.path.exists(dest_file_path):
                dest_file_path = os.path.join(dest_dir, f"{base_name}_{counter}{ext}")
                counter += 1

            # Move the file to the destination
            shutil.move(src_file_path, dest_file_path)
            print(f"Moved '{file_name}' to '{dest_file_path}'.")

def load_images_from_folder(folder_path):
    """
    Load image paths from the specified folder and convert them to base64.

    Args:
        folder_path (str): The path to the folder containing images.

    Returns:
        list: A list of image HTML `<img>` tags with base64-encoded images.
    """
    image_html = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):  # Ensure valid image formats
            # Get absolute path to the image file
            image_path = os.path.join(folder_path, filename)
            
            # Open the image file and convert it to base64
            with open(image_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Create an HTML img tag with base64 image and reduce size to fit the layout
            img_tag = f'<div style="flex: 1; margin: 10px; text-align: center; width: 45%;">' \
                      f'<img src="data:image/png;base64,{encoded_image}" alt="Chart" ' \
                      f'style="max-width: 90%; height: auto; border: 1px solid #BDC3C7; border-radius: 5px; padding: 10px;"/>' \
                      f'</div>'
            image_html.append(img_tag)
    
    # Group the images in rows of 2 (2 per row)
    grouped_images_html = []
    for i in range(0, len(image_html), 2):
        grouped_images_html.append(
            f'<div style="display: flex; justify-content: space-between; flex-wrap: wrap;">' +
            ''.join(image_html[i:i+2]) +
            '</div>'  # Wrap 2 images per row
        )
    
    return grouped_images_html

def generate_color_palette(base_color='#3498db'):
    """
    Generate a sophisticated color palette from a base color.
    
    Args:
        base_color (str): Hex color code to generate palette from
    
    Returns:
        dict: Comprehensive color palette
    """
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    def rgb_to_hex(rgb):
        return '#{:02x}{:02x}{:02x}'.format(max(0, min(255, int(rgb[0]))), 
                                            max(0, min(255, int(rgb[1]))), 
                                            max(0, min(255, int(rgb[2]))))
    
    def adjust_color(rgb, h_shift=0, s_shift=0, v_shift=0):
        h, s, v = colorsys.rgb_to_hsv(rgb[0]/255.0, rgb[1]/255.0, rgb[2]/255.0)
        h = (h + h_shift) % 1.0
        s = max(0, min(1, s + s_shift))
        v = max(0, min(1, v + v_shift))
        return [int(x * 255) for x in colorsys.hsv_to_rgb(h, s, v)]
    
    base_rgb = hex_to_rgb(base_color)
    
    return {
        'primary': base_color,
        'primary_light': rgb_to_hex(adjust_color(base_rgb, v_shift=0.2)),
        'primary_dark': rgb_to_hex(adjust_color(base_rgb, v_shift=-0.2)),
        'secondary': rgb_to_hex(adjust_color(base_rgb, h_shift=0.5)),
        'accent': rgb_to_hex(adjust_color(base_rgb, h_shift=0.25)),
        'background': '#F4F6F7',
        'text_primary': '#2C3E50',
        'text_secondary': '#34495E',
        'border': '#ECF0F1'
    }


def save_to_pdf(markdown_content, file_name="output.pdf", charts_folder="charts", color_palette=None):
    """
        Save Markdown content to a PDF file with enhanced visual design.
    """
    # Generate color palette if not provided
    if color_palette is None:
        color_palette = generate_color_palette()
    
    # Enhanced Markdown conversion
    html_content = markdown2.markdown(
        markdown_content, 
        extras=[
            'fenced-code-blocks', 
            'tables', 
            'metadata', 
            'footnotes', 
            'header-ids', 
            'wiki-tables'
        ]
    )
    
    # Generate current date
    current_date = datetime.now().strftime("%B %d, %Y")
    
    # CSS
    css_content = f"""
    /* Import modern, clean fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Roboto+Slab:wght@300;400;700&display=swap');

    /* Page and Print Styling */
    @page {{
        size: A4;
        margin: 15mm;
        background-color: {color_palette['background']};
        @top-center {{
            content: string(document-title);
            color: {color_palette['text_secondary']};
        }}
        @bottom-center {{
            content: "Page " counter(page) " of " counter(pages);
            color: {color_palette['text_secondary']};
        }}
    }}

    :root {{
        --primary-color: {color_palette['primary']};
        --primary-light: {color_palette['primary_light']};
        --primary-dark: {color_palette['primary_dark']};
        --secondary-color: {color_palette['secondary']};
        --accent-color: {color_palette['accent']};
        --text-primary: {color_palette['text_primary']};
        --text-secondary: {color_palette['text_secondary']};
        --background-color: {color_palette['background']};
        --border-color: {color_palette['border']};
    }}

    /* Global Reset and Base Styling */
    body {{
        font-family: 'Poppins', sans-serif;
        line-height: 1.6;
        color: var(--text-primary);
        background-color: var(--background-color);
        max-width: 700px;
        margin: 0 auto;
        padding: 20px;
        font-size: 10.5pt;
        text-rendering: optimizeLegibility;
    }}

    /* Sophisticated Header Styles */
    h1, h2, h3, h4, h5, h6 {{
        font-family: 'Roboto Slab', serif;
        color: var(--primary-dark);
        margin-top: 1.5em;
        font-weight: 700;
        position: relative;
    }}

    h1 {{
        font-size: 22pt;
        border-bottom: 3px solid var(--primary-color);
        padding-bottom: 10px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}

    h2 {{
        font-size: 16pt;
        padding-left: 15px;
        border-left: 5px solid var(--accent-color);
        color: var(--primary-dark);
    }}

    h3 {{
        font-size: 13pt;
        color: var(--secondary-color);
    }}

    /* Enhanced Code Blocks */
    pre {{
        background-color: #f8f9fa;
        border: 1px solid var(--border-color);
        border-radius: 6px;
        padding: 15px;
        overflow-x: auto;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }}

    code {{
        font-family: 'Courier New', monospace;
        background-color: #f1f3f4;
        border-radius: 4px;
        padding: 2px 4px;
        font-size: 9pt;
    }}

    /* Table Styling with Modern Approach */
    table {{
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        margin: 1.5em 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border-radius: 8px;
        overflow: hidden;
    }}

    th {{
        background-color: var(--primary-light);
        color: white;
        padding: 12px;
        text-transform: uppercase;
        font-size: 9pt;
        letter-spacing: 0.5px;
    }}

    td {{
        border: 1px solid var(--border-color);
        padding: 10px;
        background-color: white;
    }}

    /* List Styling */
    ul, ol {{
        padding-left: 30px;
    }}

    li {{
        margin-bottom: 0.5em;
        position: relative;
    }}


    /* Blockquote Styling */
    blockquote {{
        border-left: 4px solid var(--primary-color);
        padding-left: 15px;
        color: var(--text-secondary);
        font-style: italic;
        margin: 1.5em 0;
        background-color: rgba(52, 152, 219, 0.05);
    }}

    /* Document Header */
    .document-header {{
        text-align: center;
        margin-bottom: 30px;
        padding-bottom: 15px;
        border-bottom: 2px solid var(--primary-color);
    }}

    .document-header h1 {{
        border: none;
        color: var(--primary-dark);
        margin-bottom: 10px;
    }}

    .metadata {{
        color: var(--text-secondary);
        font-size: 9pt;
    }}

    /* Footer */
    footer {{
        margin-top: 30px;
        text-align: center;
        color: var(--text-secondary);
        border-top: 1px solid var(--border-color);
        padding-top: 10px;
    }}
    """
    
    # Load images from the charts folder
    images_html = "".join(load_images_from_folder(charts_folder))
    
    header_html = f"""
    <div class="document-header">
        <h1></h1>
        <div class="metadata">
            <span>Generated on: {current_date}</span>
        </div>
    </div>
    """
    
    footer_html = """
    <footer>
        <p>Generated with Markdown-to-PDF Converter</p>
    </footer>
    """
    
    # Full HTML composition
    full_html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title></title>
    </head>
    <body>
        {header_html}
        {html_content}
        <h2 style="color:#34495E; margin-top: 30px;">Relevant Charts Generated:</h2>
        {images_html}
        {footer_html}
    </body>
    </html>
    """
    
    try:
        HTML(string=full_html_content).write_pdf(
            file_name, 
            stylesheets=[CSS(string=css_content)]
        )
        print(f"✅ PDF successfully generated: {file_name}")
    except Exception as e:
        with open("output.md", "w") as md_file:
            md_file.write(markdown_content)
        print(f"❌ Error generating PDF: {e}")
    


import os

def append_to_file(filename, text):
    """
    Appends the given text to a .txt file.
    If the file exists, it is deleted before appending.

    Args:
        filename (str): The name of the .txt file.
        text (str): The text to append.
    """
    try:
        # Open file in append mode (it creates the file if not present)
        with open(filename, 'a') as file:
            file.write(text + '\n')  # Append the text followed by a newline
        print(f"Text appended to {filename}")
    except Exception as e:
        print(f"An error occurred: {e}")