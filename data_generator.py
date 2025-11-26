import json
import random
from faker import Faker

fake = Faker()

FUNCTIONAL_AREAS = [
    "Management", "Operations", "Marketing", "Sales", "Finance/Accounting",
    "Human Resources", "Information Technology (IT)", "Research and Development (R&D)",
    "Legal", "Customer Service/Customer Experience"
]

SKILLS = {
    "Management": "Leadership, Strategic Planning, Decision Making, Team Management, Budgeting",
    "Operations": "Process Optimization, Production Planning, Quality Control, Supply Chain Management",
    "Marketing": "Brand Management, Digital Marketing, Market Research, Campaign Management",
    "Sales": "Client Relationship Management, Negotiation, Sales Strategy, Retail Management",
    "Finance/Accounting": "Financial Reporting, Auditing, Budgeting, Tax Compliance, Forecasting",
    "Human Resources": "Recruitment, Employee Engagement, Labor Law, HRIS, Performance Management",
    "Information Technology (IT)": "Systems Administration, Network Security, Cloud Infrastructure, ERP Management",
    "Research and Development (R&D)": "Product Innovation, Prototyping, Testing, Project Management",
    "Legal": "Contract Law, Compliance, Risk Management, Regulatory Affairs",
    "Customer Service/Customer Experience": "Customer Support, Feedback Analysis, CRM, Service Quality"
}

JOB_TITLES = {
    "Management": ["General Manager", "Operations Director", "Executive Director"],
    "Operations": ["Operations Manager", "Production Supervisor", "Logistics Coordinator"],
    "Marketing": ["Marketing Manager", "Brand Strategist", "Digital Marketing Specialist"],
    "Sales": ["Sales Executive", "Retail Manager", "Wholesale Account Manager"],
    "Finance/Accounting": ["Financial Analyst", "Cost Accountant", "Controller"],
    "Human Resources": ["HR Manager", "Compensation and Benefits Specialist", "Recruitment Lead"],
    "Information Technology (IT)": ["IT Manager", "Systems Administrator", "Data Analyst"],
    "Research and Development (R&D)": ["R&D Engineer", "Product Developer", "Innovation Specialist"],
    "Legal": ["Corporate Lawyer", "Compliance Officer", "Legal Counsel"],
    "Customer Service/Customer Experience": ["Customer Service Manager", "Client Relations Specialist",
                                             "Experience Coordinator"]
}

def generate_education():
    degrees = ["Ph.D.", "M.S.", "B.Sc.", "MBA", "Diploma"]
    fields = ["Business Administration", "Finance", "Marketing", "Engineering", "Human Resources",
              "Computer Science", "Physics", "Chemistry", "Mechanical Engineering"]
    edus = []
    for _ in range(random.randint(2, 4)):
        degree = random.choice(degrees)
        field = random.choice(fields)
        university = f"{fake.city()} University"
        edus.append({"degree": degree, "field": field, "institution": university})
    return edus

def generate_experience(functional_area):
    roles = JOB_TITLES[functional_area]
    exp_list = []
    for _ in range(random.randint(1, 3)):
        role = random.choice(roles)
        years = f"{random.randint(2010, 2018)}–{random.randint(2019, 2025)}"
        duties = f"- Executed tasks in {functional_area.lower()} within luxury watchmaking.\n" \
                 f"- Collaborated with cross-functional teams and international partners.\n" \
                 f"- Maintained adherence to Swiss luxury standards and ISO compliance."
        exp_list.append({"role": role, "years": years, "details": duties})
    return exp_list

def generate_skills(functional_area):
    return SKILLS[functional_area]

def generate_dummy_cv(functional_area, label):
    name = fake.name()
    education = generate_education()
    experience = generate_experience(name, functional_area)
    skills = generate_skills(functional_area)

    raw_text = f"Name: {name}\nEmail: {fake.email()}\nPhone: {fake.phone_number()}\nLocation: Switzerland\n\n"
    raw_text += f"Summary:\nHighly skilled professional in {functional_area.lower()} for a Swiss luxury watch brand, "
    raw_text += "combining industry expertise with Swiss precision standards.\n\n"
    raw_text += "Experience:\n"
    for exp in experience:
        raw_text += f"Role: {exp['role']}\nYears: {exp['years']}\n{exp['details']}\n\n"
    raw_text += "Skills:\n" + skills + "\n\n"
    raw_text += "Education:\n"
    for edu in education:
        raw_text += f"{edu['degree']} in {edu['field']} from {edu['institution']}\n"

    job_title = experience[0]['role']
    return (job_title, raw_text, label)

def generate_dummy_cv_tuple(label: int):
    """
    Generates a tuple (job_title, raw_text, label) suitable for training.
    label=1: CV matches the functional area (positive sample)
    label=0: CV from unrelated area (negative sample)
    """
    target_area = random.choice(FUNCTIONAL_AREAS)

    if label == 1:
        functional_area = target_area
    else:
        unrelated_areas = [area for area in FUNCTIONAL_AREAS if area != target_area]
        functional_area = random.choice(unrelated_areas)

    name = fake.name()
    education = generate_education()
    experience = generate_experience(name, functional_area)
    skills = generate_skills(functional_area)

    job_title = experience[0]['role']

    raw_text = f"Name: {name}\nEmail: {fake.email()}\nPhone: {fake.phone_number()}\nLocation: Switzerland\n\n"
    raw_text += f"Summary:\nHighly skilled professional in {functional_area.lower()} for a Swiss luxury watch brand, "
    raw_text += "combining industry expertise with Swiss precision standards.\n\nExperience:\n"
    for exp in experience:
        raw_text += f"Role: {exp['role']}\nYears: {exp['years']}\n{exp['details']}\n\n"
    raw_text += "Skills:\n" + skills + "\n\nEducation:\n"
    for edu in education:
        raw_text += f"{edu['degree']} in {edu['field']} from {edu['institution']}\n"

    return (job_title, raw_text, label)

NUM_TRAIN = 1000
NUM_VAL = 200

dummy_train = [generate_dummy_cv_tuple(random.choice([0,1])) for _ in range(NUM_TRAIN)]
dummy_val = [generate_dummy_cv_tuple(random.choice([0,1])) for _ in range(NUM_VAL)]

def save_jsonl(filename, data):
    with open(filename, "w", encoding="utf-8") as f:
        for job_text, raw_text, label in data:
            record = {"job_title": job_text, "raw_text": raw_text, "label": label}
            f.write(json.dumps(record) + "\n")

save_jsonl("data/master_resumes_train.jsonl", dummy_train)
save_jsonl("data/master_resumes_val.jsonl", dummy_val)

print(f"Generated {NUM_TRAIN} training CVs and {NUM_VAL} validation CVs for the single luxury watch brand.")
