import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from PIL import Image
import shutil


class AnomalyDetector:
    def __init__(self, image_list):
        self.image_list = image_list
        self.features = [self.extract_resolution_features(image) for image in image_list]  
        self.kmeans = KMeans(n_clusters=2, random_state=42) 
        self.kmeans.fit(self.features)

    def extract_resolution_features(self, image):
        height, width = image.shape
        return np.array([width, height])

    def detect_anomaly(self, image):
        feature = self.extract_resolution_features(image)
        label = self.kmeans.predict([feature])[0]  
        center = self.kmeans.cluster_centers_[label]  
        distance = np.linalg.norm(feature - center)  
        
        max_distance = np.max([np.linalg.norm(f - center) for f in self.features])
        anomaly_probability = distance / max_distance
        return anomaly_probability

    def compare_similarity(self, image1, image2):
        feature1 = self.extract_resolution_features(image1)
        feature2 = self.extract_resolution_features(image2)
        
        distance = np.linalg.norm(feature1 - feature2)
        similarity = 1 / (1 + distance)  
        return similarity


