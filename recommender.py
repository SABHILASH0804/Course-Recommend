import torch
import torch.nn.functional as F
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import sys
import json
import os

sys.stdout.reconfigure(encoding='utf-8')

# Load dataset
try:
    df = pd.read_csv("Coursera.csv")
    sys.stderr.write("Dataset loaded successfully.\n")
except Exception as e:
    sys.stderr.write(f"Error loading dataset: {e}\n")
    sys.exit(1)

df = df[["Course Name", "Skills", "Course URL", "Course Rating", "Course Description"]].dropna()
df["Course Rating"] = pd.to_numeric(df["Course Rating"].replace("Not Calibrated", None), errors="coerce").fillna(0)
df["Skill Count"] = df["Skills"].apply(lambda x: len(str(x).split("  ")) if pd.notna(x) else 0)

skill_to_courses = {}
course_skills = {}

for idx, row in df.iterrows():
    skills = [s.strip().lower() for s in row["Skills"].split("  ") if s.strip()]
    course_skills[idx] = skills
    for skill in skills:
        skill_to_courses.setdefault(skill, []).append(idx)

sys.stderr.write(f"Total unique skills processed: {len(skill_to_courses)}\n")

edges = set()
for courses in skill_to_courses.values():
    for i in range(len(courses)):
        for j in range(i + 1, len(courses)):
            edges.add((courses[i], courses[j]))

sys.stderr.write(f"Total edges created: {len(edges)}\n")

edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()
course_features = torch.tensor(df[["Course Rating", "Skill Count"]].values, dtype=torch.float32)

class CourseGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(CourseGNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

data = Data(x=course_features, edge_index=edge_index)
model = CourseGNN(in_channels=2, hidden_channels=16, out_channels=1)
course_scores = None

def initialize_model():
    global course_scores
    if os.path.exists("model.pth"):
        model.load_state_dict(torch.load("model.pth"))
        model.eval()
        with torch.no_grad():
            course_scores = model(data.x, data.edge_index).squeeze()
        sys.stderr.write("Loaded pre-trained model and precomputed scores.\n")
    else:
        sys.stderr.write("❌ Error: Pre-trained model not found. Run with 'python recommender.py train' to train.\n")
        sys.exit(1)

def recommend_courses(extracted_skills, top_n=40):
    global course_scores
    sys.stderr.write(f"Received extracted skills: {extracted_skills}\n")

    extracted_skills = [skill.strip().lower() for skill in extracted_skills.split(',')]
    relevant_courses = set()
    matched_skills = set()

    for skill in extracted_skills:
        if skill in skill_to_courses:
            relevant_courses.update(skill_to_courses[skill])
            matched_skills.add(skill)
        else:
            for course_skill in skill_to_courses:
                if skill in course_skill:
                    relevant_courses.update(skill_to_courses[course_skill])
                    matched_skills.add(course_skill)

    sys.stderr.write(f"Matched Skills: {matched_skills}\n")
    sys.stderr.write(f"Total relevant courses found: {len(relevant_courses)}\n")

    if not relevant_courses:
        sys.stderr.write("No courses found for extracted skills.\n")
        return [], set(), set()

    relevant_courses_set = relevant_courses.copy()
    relevant_courses = list(relevant_courses)

    ranked_courses = sorted(relevant_courses, key=lambda idx: course_scores[idx].item(), reverse=True)

    recommended_set = set(ranked_courses[:top_n])
    recommendations = []

    for idx in ranked_courses[:top_n]:
        course_info = df.iloc[idx][["Course Name", "Course URL", "Course Rating", "Course Description"]].to_dict()
        course_info["Matched Skills"] = [skill for skill in course_skills[idx] if skill in matched_skills]
        recommendations.append(course_info)

    sys.stderr.write(f"Top {top_n} recommended courses found.\n")
    return recommendations, relevant_courses_set, recommended_set

def display_metrics(relevant_courses_set, recommended_set):
    true_positives = len(recommended_set.intersection(relevant_courses_set))
    false_positives = len(recommended_set.difference(relevant_courses_set))
    false_negatives = len(relevant_courses_set.difference(recommended_set))
    true_negatives = len(df) - (true_positives + false_positives + false_negatives)

    sys.stderr.write(f"True Positives: {true_positives}\n")
    sys.stderr.write(f"False Positives: {false_positives}\n")
    sys.stderr.write(f"False Negatives: {false_negatives}\n")
    sys.stderr.write(f"True Negatives: {true_negatives}\n")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].lower() == "train":
        def train_model(epochs=500, lr=0.07):
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            loss_fn = torch.nn.MSELoss()
            for epoch in range(epochs):
                model.train()
                optimizer.zero_grad()
                predicted_ratings = model(data.x, data.edge_index)
                loss = loss_fn(predicted_ratings, data.x[:, 0].view(-1, 1))
                loss.backward()
                optimizer.step()
                if epoch % 20 == 0:
                    sys.stderr.write(f"Epoch {epoch}, Loss: {loss.item():.4f}\n")
            torch.save(model.state_dict(), "model.pth")
            sys.stderr.write("✅ Training completed and model saved.\n")
        sys.stderr.write("Training GNN model...\n")
        train_model()
    else:
        initialize_model()
        extracted_skills = sys.argv[1]
        sys.stderr.write(f"Processing skills: {extracted_skills}\n")
        recommendations, relevant_courses_set, recommended_set = recommend_courses(extracted_skills)
        print(json.dumps(recommendations))
        display_metrics(relevant_courses_set, recommended_set)
