# IBM-cloud-project
A machine learning-based solution to classify applicants into the correct NSAP (National Social Assistance Program) scheme using demographic and socio-economic data. The model is deployed via IBM Watsonx.ai and accessed through a REST API for real-time predictions.


NSAP Scheme Classification Using Machine Learning
📌 Overview
This project aims to automate the scheme recommendation process under the National Social Assistance Programme (NSAP) by predicting the most suitable welfare scheme for applicants based on their demographic and socio-economic details. It uses a multi-class classification model developed and deployed on IBM Watsonx.ai Studio with data sourced from AI Kosh.

🎯 Problem Statement
Manually verifying applications for NSAP and assigning the correct sub-scheme is time-consuming and error-prone. Delays or misallocations often prevent deserving individuals from receiving timely financial assistance.

✅ Proposed Solution
The solution involves:
Building a machine learning model that predicts the appropriate NSAP scheme based on applicant data.
Deploying the model using IBM Watsonx.ai and IBM Cloud Assistant for real-time predictions.
Creating a user-friendly interface for inputting applicant details and retrieving predictions.

⚙️ System Requirements
Python 3.x
Jupyter Notebook
IBM Watsonx.ai account
IBM Cloud deployment setup
Internet connection
📚 Libraries Used
python
Copy
Edit
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib
🧠 Algorithm & Model
Model Used: Random Forest Classifier
Input Features: Age, Gender, Marital Status, Disability Status, Income Group, Location, etc.
Training: The model was trained on preprocessed AI Kosh data with proper encoding and balancing techniques.
Evaluation Metrics: Accuracy, F1 Score, Confusion Matrix

🚀 Deployment
Model was deployed using IBM Watsonx.ai Studio
Real-time prediction API was generated using IBM Cloud Deployment Service
User interface created with Watson Assistant for easy interaction

🧪 Testing Interface in Watsonx.ai
📈 Confusion Matrix and Accuracy Report
(Include images in your repo or as markdown if needed)

🔍 Future Scope
Integrate additional applicant data (e.g., health records, income statements)
Deploy to more regions and incorporate local languages
Use edge computing for offline accessibility in rural areas
Upgrade to advanced algorithms like XGBoost or neural networks
Add transparency via Explainable AI (XAI)

📖 References
NSAP Official Portal
AI Kosh Dataset
IBM Watsonx.ai Studio
IBM Cloud
IBM Cloud Assistant
Scikit-learn Docs
Machine Learning Mastery
Towards Data Science

📂 Folder Structure
objectivec
Copy
Edit
├── dataset/
│   └── NSAP_dataset.csv
├── model/
│   └── trained_model.pkl
├── notebook/
│   └── NSAP_Classification.ipynb
├── interface/
│   └── Watson_UI_Screenshots/
├── README.md

🙌 Acknowledgements
Special thanks to IBM SkillsBuild and Watsonx.ai Studio for providing tools and platforms for model development.
Government of India for open access to welfare scheme data via AI Kosh.
