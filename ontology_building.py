#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install pandas')
get_ipython().system('pip install matplotlib')


# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import numpy as np
from PIL import Image


# In[3]:


DATA_DIR = "/home/student/Desktop/fyp-Medpix/MedPix-2.0-main/MedPix-2-0"      

IMG_DIR = os.path.join(DATA_DIR, "images")
CASE_FILE = os.path.join(DATA_DIR, "Case_topic.json")
DESC_FILE = os.path.join(DATA_DIR, "Descriptions.json")
IMG_OVR = os.path.join(DATA_DIR, "images_overview.csv")


# In[4]:


print(IMG_DIR)
print(CASE_FILE)
print(DESC_FILE)
print(IMG_OVR)


# In[5]:


# Load case-level info
with open(CASE_FILE, "r") as f:
    case_data = json.load(f)

# Load image-level info
with open(DESC_FILE, "r") as f:
    desc_data = json.load(f)

# Create quick lookup: image_name -> description
desc_index = {item["image"]: item for item in desc_data}


# In[6]:


print(desc_index)


# In[7]:


print(json.dumps(case_data[0], indent=4))


# In[8]:


print(json.dumps(desc_data[0], indent=4))


# In[9]:


print(os.listdir(IMG_DIR)[:20])


# In[10]:


def load_image(img_name):
    path = os.path.join(IMG_DIR, img_name)
    print(os.listdir(IMG_DIR)[:20])

    return Image.open(path).convert("RGB")
def show_image(img, title="Image"):
    plt.figure(figsize=(5,5))
    plt.imshow(img)
    plt.title(title)
    plt.axis("off")
    plt.show()

img = load_image("MPX1009_synpic46283.png")
show_image(img, title="Sample Image")


# In[11]:


print("Total number of cases:", len(case_data))
print("Total number of image descriptions:", len(desc_data))
image_files = [f for f in os.listdir(IMG_DIR) if f.endswith(".png")]
print("Total PNG images:", len(image_files))


# In[12]:


ct_count = 0
mri_count = 0

for case in case_data:
    ct_count += len(case["TAC"])
    mri_count += len(case["MRI"])

print("Total CT images listed in Case_topic:", ct_count)
print("Total MRI images listed in Case_topic:", mri_count)


# In[13]:


images_per_case = []

for case in case_data:
    total_imgs = len(case["TAC"]) + len(case["MRI"])
    images_per_case.append(total_imgs)

import matplotlib.pyplot as plt

plt.hist(images_per_case, bins=15)
plt.title("Images per Case")
plt.xlabel("Number of images")
plt.ylabel("Number of cases")
plt.show()


# In[14]:


ages = []

for d in desc_data:
    age = d["Description"].get("Age", None)
    if age is not None:
        ages.append(age)

plt.hist(ages, bins=20)
plt.title("Patient Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()


# In[15]:


mod_count = {"CT":0, "MR":0}

for d in desc_data:
    mod_count[d["Type"]] += 1

plt.bar(mod_count.keys(), mod_count.values())
plt.title("Modality Distribution")
plt.ylabel("Count")
plt.show()


# In[16]:


import matplotlib.pyplot as plt
import numpy as np

findings_len = []
history_len = []
c_f=0
c_h=0
for c in case_data:
    case = c.get("Case", {})

    # Debug: print if Findings or History is missing
    if "Findings" not in case:
        print("Missing Findings in case:", case)
        c_f+=1
    if "History" not in case:
        print("Missing History in case:", case)
        c_h+=1

    # Safely get Findings and History, default to empty string if missing
    f = case.get("Findings", "")
    h = case.get("History", "")

    # Count words
    findings_len.append(len(str(f).split()))
    history_len.append(len(str(h).split()))

# Plot histogram for Findings
plt.hist(findings_len, bins=20, color='skyblue', edgecolor='black')
plt.title("Length of Findings (in words)")
plt.xlabel("Number of words")
plt.ylabel("Frequency")
plt.show()

# Plot histogram for History
plt.hist(history_len, bins=20, color='lightgreen', edgecolor='black')
plt.title("Length of History (in words)")
plt.xlabel("Number of words")
plt.ylabel("Frequency")
plt.show()

print(c_f,c_h)


# In[17]:


c = 0
for i in desc_data:
    if "Description" in i and "Caption" in i["Description"]:
        c += 1  # Count this caption
print("Total captions:", c)


# In[18]:


import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


u_id_to_caption = defaultdict(str)
for d in desc_data:
    u_id = d.get("U_id")
    caption = d.get("Description", {}).get("Caption", "")
    if caption not in u_id_to_caption[u_id]:
        u_id_to_caption[u_id] += " " + caption


findings_len = []
history_len = []
c_f = 0
c_h = 0

for i, c in enumerate(case_data):
    case = c["Case"]
    u_id = c["U_id"]

    # Handle Findings
    if "Findings" not in case or not case.get("Findings"):
        print("Missing Findings in case:", case)
        c_f += 1
        case_data[i]["Case"]["Findings"] = u_id_to_caption.get(u_id)

    # Handle History
    if "History" not in case:
        print("Missing History in case:", case)
        c_h += 1
        case_data[i]["Case"]["History"] = ""

print("Cases missing Findings:", c_f)
print("Cases missing History:", c_h)


# In[19]:


u_id_to_caption


# In[20]:


from collections import defaultdict

unique_locations = defaultdict(set)
unique_location_categories = set()

for d in desc_data:
    unique_locations[d["Location"]].add(d["Location Category"])
    unique_location_categories.add(d["Location Category"])

for k in unique_locations: print(f"{k} => {unique_locations[k]}")


# In[21]:


print(len(unique_location_categories))


# In[22]:


print(unique_location_categories)


# In[23]:


# Create a mapping from U_id to Caption for faster lookup
u_id_to_desc = {d.get("U_id"): d
                   for d in desc_data}


# In[24]:


from collections import defaultdict

key_count = defaultdict(int)

for c in case_data:
    case = c.get("Case", {})
    for key in case.keys():
        key_count[key] += 1  # Increment count if key exists

# Print the counts
for key, count in key_count.items():
    print(f"{key}: {count}")


# In[25]:


DATA_DIR = "/home/student/Desktop/fyp-Medpix/MedPix-2.0-main/MedPix-2-0"      

IMG_DIR = os.path.join(DATA_DIR, "images")
CASE_FILE = os.path.join(DATA_DIR, "Case_topic.json")
DESC_FILE = os.path.join(DATA_DIR, "Descriptions.json")
IMG_OVR = os.path.join(DATA_DIR, "images_overview.csv")


# In[26]:


def clean_text(text):
    if not isinstance(text, str):
        text = str(text)

    text = text.replace("\n", " ")   # remove newlines
    text = text.replace("\t", " ")   # remove tabs
    text = re.sub(r"\s+", " ", text) # collapse multiple spaces/tabs/newlines
    return text.strip()    


# In[86]:


# df_overall
# U_id, Location Category, Location, list(CT), list(MRI), title, history, findings, case diagnosis, Differential Diagnosis, captions(concated)

import pandas as pd
from collections import defaultdict
import re
import numpy as np
# Example: suppose case_data is your array of JSON objects
# And desc_data is your array of JSON objects containing captions

data_list = []

for c in case_data:
    case = c["Case"]
    u_id = c["U_id"]

    # Extract lists (ensure default empty list if missing)
    ct_list = [os.path.join(IMG_DIR, path+".png") for path in c["TAC"]]
    mri_list = [os.path.join(IMG_DIR, path+".png") for path in c["MRI"]]

    # Extract other fields (use empty string if missing)
    title = clean_text(case["Title"])
    history = clean_text(case["History"])
    findings = clean_text(case["Findings"])
    case_diagnosis = clean_text(case["Case Diagnosis"])
    diff_diagnosis = clean_text(case.get("Differential Diagnosis", case_diagnosis))


    # Use concatenated captions from desc_data if any
    captions = u_id_to_caption[u_id]

    # Append a row dictionary
    data_list.append({
        "U_id": u_id,
        "Location Category": u_id_to_desc[u_id]["Location Category"],
        "Location": u_id_to_desc[u_id]["Location"],
        "CT_image_paths": ct_list,
        "MRI_image_paths": mri_list,
        "title": title,
        "history": history,
        "findings": findings,
        "case diagnosis": case_diagnosis,
        "Differential Diagnosis": diff_diagnosis,
        "captions": captions.strip()
    })

# Step 3: Construct DataFrame
df_overall = pd.DataFrame(data_list)


df_overall["combined_text"] = np.where(
    df_overall["findings"] != df_overall["captions"],
    df_overall["findings"] + " " + df_overall["captions"],
    df_overall["findings"]
)

# Optional: view the DataFrame
print(df_overall.head())


# In[87]:


df_overall["findings"][0]


# In[88]:


df_overall["CT_image_paths"][0]


# In[89]:


df_overall.to_csv("/home/student/Desktop/fyp-Medpix/data/df_overall.csv")


# In[90]:


# Dictionary to store all the new dataframes
dfs_by_category = {}

for category in df_overall["Location Category"].unique():
    dfs_by_category[category] = df_overall[df_overall["Location Category"] == category]


# In[91]:


for cat, df_cat in dfs_by_category.items():
    print(cat, df_cat.shape)


# In[92]:


import os

output_dir = "/home/student/Desktop/fyp-Medpix/data/split_by_location_category"
os.makedirs(output_dir, exist_ok=True)

for category, df_cat in dfs_by_category.items():
    safe_name = str(category).replace(" ", "_").replace("/", "-")
    df_cat.to_csv(f"{output_dir}/{safe_name}.csv", index=False)


# In[93]:


import os

output_dir = "/home/student/Desktop/fyp-Medpix/data/split_by_location_category"
os.makedirs(output_dir, exist_ok=True)

for category, df_cat in dfs_by_category.items():
    safe_name = str(category).replace(" ", "_").replace("/", "-")
    df_cat.to_csv(f"{output_dir}/{safe_name}.csv", index=False)


# In[35]:


get_ipython().system('pip install radgraph')


# In[36]:


df_overall.columns


# In[37]:


from radgraph import RadGraph

# Load RadGraph model on CPU
model = RadGraph()
model.device = "cpu"
model.model.to("cpu")

# Function to extract RadGraph entities
def extract_radgraph_entities(text):
    if not text or text.strip() == "":
        return {"anatomy": [], "observation": [], "uncertain": [], "relations": []}

    result = model(text)  # inference on CPU
    entities = result.get("entities", {})
    relations = result.get("relations", [])

    anatomy = []
    observation = []
    uncertain = []
    rel_list = []

    for ent_id, ent in entities.items():
        label = ent.get("label")
        if label == "ANAT-DP":
            anatomy.append(ent["text"])
        elif label == "OBS-DP":
            observation.append(ent["text"])
        elif label == "UNCERTAIN":
            uncertain.append(ent["text"])

    for rel in relations:
        head = entities.get(rel["head"], {}).get("text", "")
        tail = entities.get(rel["tail"], {}).get("text", "")
        rel_list.append((head, rel["type"], tail))

    return {
        "anatomy": list(set(anatomy)),
        "observation": list(set(observation)),
        "uncertain": list(set(uncertain)),
        "relations": rel_list
    }

# Apply RadGraph on your findings
findings_list = df_overall["findings"].tolist()
radgraph_output = [extract_radgraph_entities(text) for text in findings_list]

# Add RadGraph columns to df_overall
# df_overall["anatomy_terms"] = [x["anatomy"] for x in radgraph_output]
# df_overall["observation_terms"] = [x["observation"] for x in radgraph_output]
# df_overall["uncertain_terms"] = [x["uncertain"] for x in radgraph_output]
# df_overall["relations"] = [x["relations"] for x in radgraph_output]


# In[38]:


print(radgraph_output)
print(findings_list)


# In[39]:


from radgraph import RadGraph

model = RadGraph()
model.device = "cpu"
model.model.to("cpu")

sample_text = df_overall['findings'].iloc[0]
result = model(sample_text)
print(result)


# In[40]:


from radgraph import RadGraph
import pandas as pd
import re

# ----------------------------
# Load RadGraph model on CPU
# ----------------------------
model = RadGraph()
model.device = "cpu"
model.model.to("cpu")

# ----------------------------
# RadGraph extraction function
# ----------------------------
def extract_radgraph_entities(text):
    if not text:
        return {"anatomy": [], "observation": [], "uncertain": [], "relations": []}

    result = model(text)        # Run inference
    result = result["0"]        # Extract inner dict

    entity_dict = result["entities"]

    anatomy = []
    observation = []
    uncertain = []
    relations_list = []

    # ---- Parse entities ----
    for ent_id, ent in entity_dict.items():
        token = ent["tokens"]
        label = ent["label"]

        if label == "Anatomy::definitely present":
            anatomy.append(token)

        elif label == "Observation::definitely present":
            observation.append(token)

        elif "uncertain" in label.lower():
            uncertain.append(token)

        # Parse relations for each entity
        for rel in ent["relations"]:
            rel_type = rel[0]
            tail_id = rel[1]
            tail_text = entity_dict[tail_id]["tokens"] if tail_id in entity_dict else None

            relations_list.append((token, rel_type, tail_text))

    return {
        "anatomy": list(set(anatomy)),
        "observation": list(set(observation)),
        "uncertain": list(set(uncertain)),
        "relations": relations_list
    }

# ----------------------------
# Apply RadGraph to entire column
# ----------------------------

temp_df = pd.DataFrame({
    "text": df_overall["findings"]+ " " + df_overall["captions"]
})

radgraph_output = temp_df["text"].apply(extract_radgraph_entities)

print(radgraph_output)


# In[41]:


print(json.dumps(radgraph_output[0],indent=4))


# In[42]:


radgraph_output.to_json("/home/student/Desktop/fyp-Medpix/data/radgraph_output.json", orient="values", indent=4)


# In[43]:


import json
from collections import defaultdict

# ---------------------------------------
# Initialize your final graph structure
# ---------------------------------------
ontology = {
    "Abdomen": {"anatomy": [], "observation": [], "relations": []},
    "Head": {"anatomy": [], "observation": [], "relations": []},
    "Reproductive and Urinary System": {"anatomy": [], "observation": [], "relations": []},
    "Spine and Muscles": {"anatomy": [], "observation": [], "relations": []},
    "Thorax": {"anatomy": [], "observation": [], "relations": []}
}

# Convert lists to sets internally to avoid duplicates
graph_sets = {
    k: {
        "anatomy": set(),
        "observation": set(),
        "relations": set()
    }
    for k in ontology
}

# ---------------------------------------
# Loop over each record in df_overall
# ---------------------------------------
for idx, row in df_overall.iterrows():

    # Get the correct category (this is your first-level key)
    category = row["Location Category"]
    rad = radgraph_output[idx]

    for a in rad.get("anatomy", []):
        graph_sets[category]["anatomy"].add(a)

    for o in rad.get("observation", []):
        graph_sets[category]["observation"].add(o)

    for rel in rad.get("relations", []):
        # Convert relation list → tuple ("thickened", "located_at", "wall")
        graph_sets[category]["relations"].add(tuple(rel))


for cat in ontology:
    ontology[cat]["anatomy"] = list(graph_sets[cat]["anatomy"])
    ontology[cat]["observation"] = list(graph_sets[cat]["observation"])
    ontology[cat]["relations"] = [list(r) for r in graph_sets[cat]["relations"]]


# ---------------------------------------
# Save as JSON if desired
# ---------------------------------------
with open("/home/student/Desktop/fyp-Medpix/data/ontology.json", "w") as f:
    json.dump(ontology, f, indent=4)


# In[44]:


get_ipython().system('pip install networkx')


# In[45]:


get_ipython().system('pip install scipy')


# In[50]:


df_overall.head()


# In[55]:


import json

# Load ontology
with open("ontology.json", "r") as f:
    ontology = json.load(f)

unique_keywords = set()

for system, categories in ontology.items():
    # Add anatomy keywords
    if "anatomy" in categories:
        unique_keywords.update(categories["anatomy"])

    # Add observation keywords
    if "observation" in categories:
        unique_keywords.update(categories["observation"])

# Convert to sorted list (optional)
unique_keywords_list = sorted(unique_keywords)

print(unique_keywords_list)


# In[56]:


len(unique_keywords_list)


# In[57]:


for word in unique_keywords_list:
    if '%' in word: print(word)


# In[73]:


import re

for word in unique_keywords_list:
    if re.search(r"\d+mm", word):
        print(word)
for word in unique_keywords_list:
    if word not in ('L', 'CMV', 'MCL', 'LV') and re.fullmatch(r"^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$", word):
        print(word)


# In[71]:


for word in unique_keywords_list:
    if re.search(r"\d+th", word):
        print(word)
for word in unique_keywords_list:
    if re.search(r"\d+rd", word):
        print(word)
for word in unique_keywords_list:
    if re.search(r"\d+st", word):
        print(word)
for word in unique_keywords_list:
    if re.search(r"\d+nd", word):
        print(word)
for word in unique_keywords_list:
    if re.fullmatch(r"\d+", word):
        print(word)
for word in unique_keywords_list:
    if '/' in word and not('T' in word or 'S'  in word or 'L' in word)and re.search(r'\d+', word): print(word)


# In[82]:


import re

# Roman numeral regex
roman_pattern = r"^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$"

remove_list = []

for word in unique_keywords_list:

    # --- Case 1: contains % ---
    if '%' in word:
        remove_list.append(word)
        continue

    # --- Case 2: contains Nmm ---
    if re.search(r"\d+mm", word) or "cm" in word:
        remove_list.append(word)
        continue

    # --- Case 2b: valid Roman numeral (except excluded words) ---
    if word not in ('L', 'CMV', 'MCL', 'LV') and re.fullmatch(roman_pattern, word):
        remove_list.append(word)
        continue

    # --- Case 3: ordinal numbers ---
    if (re.search(r"\d+th", word) or
        re.search(r"\d+rd", word) or
        re.search(r"\d+st", word) or
        re.search(r"\d+nd", word)):
        remove_list.append(word)
        continue

    # --- Case 3b: pure digits (e.g., "12") ---
    if re.fullmatch(r"\d+", word):
        remove_list.append(word)
        continue

    # --- Case 3c: fraction but not containing T/S/L ---
    if '/' in word and not any(x in word for x in ('T', 'S', 'L')) and re.search(r'\d+', word):
        remove_list.append(word)
        continue


# --- Final filtered list ---
filtered_keywords = [w for w in unique_keywords_list if w not in remove_list]

print("Removed:", sorted(remove_list))
print("\nFiltered:", sorted(filtered_keywords))


# In[105]:


# import re
# from itertools import combinations

# # Create mapping
# word_to_idx = {word: idx for idx, word in enumerate(unique_keywords_list)}
# N = len(unique_keywords_list)
# matrices = {cat: [[0]*N for _ in range(N)] for cat in dfs_by_category}

# # Precompile regex for each word (word-boundary match)
# patterns = {w: re.compile(rf"\b{re.escape(w)}\b", flags=re.IGNORECASE) for w in unique_keywords_list}


# for category, df_cat in dfs_by_category.items():
#     matrix = matrices[category]

#     for text in df_cat["combined_text"]:
#         found_words = []

#         for word, pattern in patterns.items():
#             if pattern.search(text):
#                 found_words.append(word)

#         for w1, w2 in combinations(found_words, 2):
#             i, j = word_to_idx[w1], word_to_idx[w2]
#             matrix[i][j] += 1
#             matrix[j][i] += 1    # symmetric matrix



import re
from itertools import combinations

# Normalize keywords to lowercase
unique_keywords_lower = [w.lower() for w in unique_keywords_list]

# Mapping (lowercase)
word_to_idx = {word: idx for idx, word in enumerate(unique_keywords_lower)}
N = len(unique_keywords_lower)

# Create matrices
matrices = {cat: [[0]*N for _ in range(N)] for cat in dfs_by_category}

# Precompile regex patterns (case-insensitive)
patterns = {
    w: re.compile(rf"\b{re.escape(w)}\b", flags=re.IGNORECASE)
    for w in unique_keywords_lower
}

for category, df_cat in dfs_by_category.items():
    matrix = matrices[category]

    for text in df_cat["combined_text"]:

        # Lowercase text for faster scanning (optional but helpful)
        text_lower = text.lower()

        found_words = []

        # Check presence case-insensitively
        for word, pattern in patterns.items():
            if pattern.search(text_lower):   # already lowercase
                found_words.append(word)

        # Update co-occurrences
        for w1, w2 in combinations(found_words, 2):
            i, j = word_to_idx[w1], word_to_idx[w2]
            matrix[i][j] += 1
            matrix[j][i] += 1   # symmetric


# In[106]:


for category, matrix in matrices.items():
    df_matrix = pd.DataFrame(matrix,
                             index=unique_keywords_list,
                             columns=unique_keywords_list)
    safe_name = str(category).replace(" ", "_").replace("/", "-")
    output_name = f"{safe_name}_matrix.csv"
    df_matrix.to_csv('/home/student/Desktop/fyp-Medpix/data/split_by_location_category_matrices/'+output_name, index=True)

    print(f"Saved: {output_name}")


# In[ ]:
