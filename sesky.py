from flask import Flask, request, render_template, jsonify
import os
import zipfile
from pathlib import Path
from FinalWorking import CellAnalyzer  # Import your CellAnalyzer class
import glob

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "web_output"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Ensure directories exist
Path(UPLOAD_FOLDER).mkdir(exist_ok=True)
Path(OUTPUT_FOLDER).mkdir(exist_ok=True)

def convert_paths_to_strings(data):
    if isinstance(data, dict):
        return {key: convert_paths_to_strings(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_paths_to_strings(item) for item in data]
    elif isinstance(data, Path):
        return str(data)
    return data

@app.route('/')
def home():
    return render_template('index2.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'file' not in request.files:
        return render_template('index2.html', error="No file uploaded")
    
    file = request.files['file']
    if file.filename == '':
        return render_template('index2.html', error="No file selected")
    folders = [os.path.basename(folder) for folder in glob.glob(f"{OUTPUT_FOLDER}/*")]
    return render_template('results.html',message="Analysis complete!",folders=folders)
    # Save uploaded file
    upload_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(upload_path)

    # Unzip if it's a ZIP file
    if file.filename.endswith('.zip'):
        with zipfile.ZipFile(upload_path, 'r') as zip_ref:
            unzip_dir = os.path.join(UPLOAD_FOLDER, file.filename.split('.')[0])
            zip_ref.extractall(unzip_dir)
            os.remove(upload_path)  # Remove ZIP after extraction
            input_folder = unzip_dir
    else:
        input_folder = UPLOAD_FOLDER

    # Analyze using CellAnalyzer
    analyzer = CellAnalyzer(output_dir=OUTPUT_FOLDER)
    results, stats, df = analyzer.run_analysis(input_folder)

    # Prepare data for rendering
    visualizations_dir = Path(OUTPUT_FOLDER) / f"visualizations_{analyzer.timestamp}"
    visualizations = [str(img) for img in visualizations_dir.glob("*.png")]
    csv_path = str(Path(OUTPUT_FOLDER) / f"cell_analysis_{analyzer.timestamp}.csv")
    statistics_path = str(Path(OUTPUT_FOLDER) / f"statistics_{analyzer.timestamp}.txt")

    # Read statistics content
    with open(statistics_path, 'r') as f:
        statistics_content = f.read()

    return render_template(
        'results.html',
        message="Analysis complete!",
        visualizations=visualizations,
        csv_path=csv_path,
        statistics=statistics_content
    )

if __name__ == "__main__":
    app.run(debug=True)
