from flask import Flask, request, jsonify, render_template
import subprocess
import os
import json

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_resume():
    if 'resume' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['resume']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    
    try:
        # Extract skills using extract_skills.py
        result = subprocess.run(['python', 'extract_skills.py', file_path], capture_output=True, text=True, check=True)
        
        if result.returncode != 0:
            print(f"\u274c Error in extract_skills.py: {result.stderr.strip()}")
            return jsonify({'error': 'Failed to extract skills'}), 500
        
        extracted_skills = json.loads(result.stdout).get("skills", [])
        
        if not extracted_skills:
            return jsonify({'error': 'No skills extracted'}), 500
        
        print(f"\u2705 Extracted Skills: {extracted_skills}")  # Debugging output
        
        # Convert skills list to a string format that recommender.py can handle
        extracted_skills_str = ",".join(extracted_skills)
        
        # Run recommender.py with extracted skills
        recommender_result = subprocess.run(['python', 'recommender.py', extracted_skills_str], capture_output=True, text=True)
        
        if recommender_result.returncode != 0:
            print(f"\u274c Error in recommender.py: {recommender_result.stderr.strip()}")
            return jsonify({'error': 'Failed to fetch recommendations'}), 500
        
        try:
            recommendations = json.loads(recommender_result.stdout)
            top_40 = recommendations[:5]  # Limit to top 40

        except json.JSONDecodeError:
            print(f"\u274c Invalid JSON response from recommender.py: {recommender_result.stdout.strip()}")
            return jsonify({'error': 'Invalid response from recommender.py'}), 500

        print(f"\u2705 Recommendations: {json.dumps(top_40, indent=2)}")  # Debugging output
        
        return jsonify({'skills': extracted_skills, 'recommendations': top_40})
    
    except subprocess.CalledProcessError as e:
        print(f"\u274c Subprocess Error: {e}")
        return jsonify({'error': f'Error executing script: {e}'}), 500
    except Exception as e:
        print(f"\u274c Unexpected Server Error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
