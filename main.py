import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Step 1: Load an Expanded Dinosaur Dataset
data = {
    "Name": [
        "Tyrannosaurus Rex", "Brachiosaurus", "Velociraptor", "Triceratops", "Stegosaurus",
        "Diplodocus", "Spinosaurus", "Ankylosaurus", "Allosaurus", "Iguanodon",
        "Pteranodon", "Parasaurolophus", "Carnotaurus", "Apatosaurus", "Gallimimus",
        "Compsognathus", "Argentinosaurus", "Microraptor", "Therizinosaurus", "Quetzalcoatlus",
        "Mosasaurus", "Pachycephalosaurus", "Deinonychus", "Edmontosaurus", "Kentrosaurus"
    ],
    "Diet": [
        "Carnivore", "Herbivore", "Carnivore", "Herbivore", "Herbivore",
        "Herbivore", "Carnivore", "Herbivore", "Carnivore", "Herbivore",
        "Carnivore", "Herbivore", "Carnivore", "Herbivore", "Omnivore",
        "Carnivore", "Herbivore", "Carnivore", "Herbivore", "Carnivore",
        "Carnivore", "Herbivore", "Carnivore", "Herbivore", "Herbivore"
    ],
    "Length_m": [
        12.3, 22.0, 1.8, 9.0, 7.0,
        27.0, 15.0, 6.0, 12.0, 10.0,
        6.0, 9.5, 8.0, 23.0, 6.5,
        1.0, 35.0, 0.8, 10.0, 11.0,
        17.0, 4.5, 3.0, 12.5, 5.5
    ],
    "Weight_t": [
        8.4, 56.3, 0.015, 6.0, 3.5,
        25.0, 8.0, 6.0, 2.5, 4.0,
        0.08, 5.5, 1.5, 27.0, 0.3,
        0.001, 77.0, 0.003, 5.0, 0.25,
        15.0, 2.0, 0.07, 4.5, 2.5
    ],
    "Period": [
        "Cretaceous", "Jurassic", "Cretaceous", "Cretaceous", "Jurassic",
        "Jurassic", "Cretaceous", "Cretaceous", "Jurassic", "Cretaceous",
        "Cretaceous", "Cretaceous", "Cretaceous", "Jurassic", "Cretaceous",
        "Jurassic", "Cretaceous", "Cretaceous", "Cretaceous", "Cretaceous",
        "Cretaceous", "Cretaceous", "Cretaceous", "Cretaceous", "Jurassic"
    ],
    "Habitat": [
        "Land", "Land", "Land", "Land", "Land",
        "Land", "Land", "Land", "Land", "Land",
        "Air", "Land", "Land", "Land", "Land",
        "Land", "Land", "Air", "Land", "Air",
        "Water", "Land", "Land", "Land", "Land"
    ]
}

df = pd.DataFrame(data)

# Step 2: Preprocess the Dataset
# Convert categorical features to numerical
df['Diet_Num'] = df['Diet'].map({'Carnivore': 0, 'Herbivore': 1, 'Omnivore': 2})
df['Period_Num'] = df['Period'].map({'Triassic': 0, 'Jurassic': 1, 'Cretaceous': 2})
df['Habitat_Num'] = df['Habitat'].map({'Land': 0, 'Water': 1, 'Air': 2})

print("Dataset Preview:\n", df.head())

# Step 3: Exploratory Data Analysis
# Dinosaur Size by Diet
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x="Diet", y="Length_m", palette="coolwarm")
plt.title("Dinosaur Lengths by Diet")
plt.xlabel("Diet")
plt.ylabel("Length (meters)")
plt.show()

# Step 4: Clustering Dinosaurs by Size and Weight
# Normalize the data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[['Length_m', 'Weight_t']])

# Apply KMeans Clustering
kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_features)

# Visualize Clusters
plt.figure(figsize=(12, 6))
sns.scatterplot(data=df, x="Length_m", y="Weight_t", hue="Cluster", size="Weight_t", palette="viridis", sizes=(20, 200))
plt.title("Dinosaur Clusters by Size and Weight")
plt.xlabel("Length (meters)")
plt.ylabel("Weight (tons)")
plt.legend()
plt.show()

# Step 5: Analysis by Period
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x="Period", hue="Diet", palette="pastel")
plt.title("Dinosaur Diet Distribution by Period")
plt.xlabel("Time Period")
plt.ylabel("Count")
plt.legend(title="Diet")
plt.show()

# Step 6: Save the Results
df.to_csv("expanded_dinosaur_analysis.csv", index=False)
print("Results saved to 'expanded_dinosaur_analysis.csv'.")
