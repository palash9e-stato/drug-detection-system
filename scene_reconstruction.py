
import numpy as np
import cv2
import PIL.Image
import plotly.graph_objects as go
from transformers import pipeline
import streamlit as st

class SceneReconstructor:
    def __init__(self):
        self.depth_estimator = None
        
    def load_model(self):
        if self.depth_estimator is None:
            # lightweight depth model
            self.depth_estimator = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf")
            
    def process_image(self, image_input):
        """
        Takes a PIL Image or numpy array.
        Returns a Plotly Figure (3D Surface Mesh).
        """
        self.load_model()
        
        if isinstance(image_input, np.ndarray):
            image = PIL.Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
        else:
            image = image_input
            
        # Get Depth
        depth = self.depth_estimator(image)["depth"]
        depth = np.array(depth)
        
        # High-Fidelity settings
        # Resize to a higher resolution for better surface detail
        target_size = (500, 500) 
        
        img_resized = image.resize(target_size)
        depth_resized = cv2.resize(depth, target_size)
        
        # Create Surface Mesh
        width, height = target_size
        
        # Prepare color scale from image
        # For go.Surface, surfacecolor needs to be mapped or we can use the image as texture
        # Plotly surfacecolor expects an array of values to map to colorscale, 
        # OR we can just use the depth as Z and correct aspect ratio.
        # But to look "exactly like the image", we need the texture.
        # Plotly is tricky with direct RGB texture on Surface. 
        # A trick is to use surfacecolor with a custom colorscale or just use the z-height with a generic texture.
        # HOWEVER, the user wants "Maximum Accuracy".
        # The best way in Plotly for "Photo-realistic" 3D is actually Surface with `surfacecolor`.
        
        # We need to construct the surfacecolor array matching the image pixels.
        # Image is (500, 500, 3).
        
        # Convert image to a format Plotly can accept for coloring?
        # Actually go.Surface supports `surfacecolor` as an array of numerical values, usually for heatmaps.
        # Mapping true RGB texture to go.Surface is complex.
        # Alternative: return to Point Cloud but DENSE (one point per pixel).
        # A 500x500 point cloud is 250,000 points. Modern browsers can handle this.
        # Let's try DENSE Point Cloud first as it guarantees exact color match.
        
        # Update: User wants "Exactly like 2d image". 
        # A dense point cloud (1-to-1 pixel mapping) is the most accurate representation of the 2D image in 3D space.
        # Let's try 300x300 first to be safe on performance, but render as points.
        
        # BETTER APPROACH FOR "HOLOGRAM":
        # Keep Surface for the "Solid" look, but use a grey colorscale if texture fails, 
        # OR use the Dense Point Cloud with small markers.
        
        # Let's allow the user to see the "Structure" (Depth) clearly.
        # I will use a high-res Surface with a nice color map (e.g., Plasma) to show depth, OR
        # attempt the RGB mapping if possible.
        # Actually, for "Drug Detection", seeing the SHAPE is often more important than the texture.
        # But user said "exactly like 2d image".
        
        # Let's go with a very dense Point Cloud (Scatter3d) but optimized.
        # 400x400 = 160k points.
        
        target_size = (400, 400)
        img_small = image.resize(target_size)
        depth_small = cv2.resize(depth, target_size)
        
        x, y = np.meshgrid(np.arange(target_size[0]), np.arange(target_size[1]))
        x = x.flatten()
        y = y.flatten()
        z = depth_small.flatten()
        
        # Color array
        colors = np.array(img_small).reshape(-1, 3) / 255.0
        
        fig = go.Figure(data=[go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(
                size=2, # Smaller dots for smoother look
                color=colors,
                opacity=1.0 # Solid
            )
        )])
        
        fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                aspectratio=dict(x=1, y=1, z=0.5),
                camera=dict(eye=dict(x=0, y=0, z=2.5)) # Top-downish view initially
            ),
            margin=dict(l=0, r=0, b=0, t=0),
            paper_bgcolor="black"
        )
        
        return fig
