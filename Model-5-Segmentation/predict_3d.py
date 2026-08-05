import os
import torch
import monai
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, 
    ScaleIntensityRangePercentilesd, SpatialPadd, ToTensord
)
from monai.networks.nets import UNet
from monai.inferers import sliding_window_inference
from monai.data import DataLoader

def classify_alzheimers(volume_mm3):
    # Thresholds based on general clinical hippocampal volumes
    # Note: These are simplified thresholds for demonstration purposes.
    if volume_mm3 >= 3200:
        return "Normal (No AD)", "green"
    elif volume_mm3 >= 2400:
        return "Early AD (Mild Atrophy)", "orange"
    else:
        return "Late AD (Severe Atrophy)", "red"

def create_3d_visualization(mri_volume, mask_volume, volume_mm3, diagnosis, sample_idx, script_dir):
    # mri_volume is 3D numpy array of the MRI crop
    # mask_volume is 3D numpy array of the predicted mask
    
    # 1. Mask Isosurface (The Hippocampus)
    z, y, x = np.where(mask_volume > 0)
    
    if len(x) == 0:
        print(f"Sample {sample_idx}: No hippocampus detected!")
        return

    # 2. MRI Orthogonal Slices (to show it "inside the brain")
    # We pick the center of the predicted mask to draw the slices
    center_z, center_y, center_x = int(np.mean(z)), int(np.mean(y)), int(np.mean(x))
    
    Z, Y, X = mri_volume.shape
    
    # Create coordinate grids for the slices
    # Axial Slice (Z constant)
    x_grid_axial, y_grid_axial = np.meshgrid(np.arange(X), np.arange(Y))
    z_grid_axial = np.full_like(x_grid_axial, center_z)
    slice_axial = mri_volume[center_z, :, :]
    
    # Coronal Slice (Y constant)
    x_grid_coronal, z_grid_coronal = np.meshgrid(np.arange(X), np.arange(Z))
    y_grid_coronal = np.full_like(x_grid_coronal, center_y)
    slice_coronal = mri_volume[:, center_y, :]
    
    # Sagittal Slice (X constant)
    y_grid_sagittal, z_grid_sagittal = np.meshgrid(np.arange(Y), np.arange(Z))
    x_grid_sagittal = np.full_like(y_grid_sagittal, center_x)
    slice_sagittal = mri_volume[:, :, center_x]
    
    # Build Plotly Figure
    fig = go.Figure()

    # Add the 3D Scatter for the Hippocampus Mask
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=3,
            color='cyan', 
            opacity=0.3 # Make it slightly transparent so we can see the context
        ),
        name="Hippocampus"
    ))

    # Add MRI Slices as Surfaces mapped with greyscale MRI intensities
    # Axial
    fig.add_trace(go.Surface(
        x=x_grid_axial, y=y_grid_axial, z=z_grid_axial,
        surfacecolor=slice_axial, colorscale='gray',
        showscale=False, opacity=0.8, name="Axial Slice"
    ))
    # Coronal
    fig.add_trace(go.Surface(
        x=x_grid_coronal, y=y_grid_coronal, z=z_grid_coronal,
        surfacecolor=slice_coronal, colorscale='gray',
        showscale=False, opacity=0.8, name="Coronal Slice"
    ))
    # Sagittal
    fig.add_trace(go.Surface(
        x=x_grid_sagittal, y=y_grid_sagittal, z=z_grid_sagittal,
        surfacecolor=slice_sagittal, colorscale='gray',
        showscale=False, opacity=0.8, name="Sagittal Slice"
    ))

    fig.update_layout(
        title=f"Sample {sample_idx} | Volume: {volume_mm3} mm³ | Diagnosis: {diagnosis}",
        scene=dict(
            xaxis_title='X Axis', yaxis_title='Y Axis', zaxis_title='Z Axis',
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=50)
    )
    
    output_html = os.path.join(script_dir, f"3D_Hippocampus_Report_Sample_{sample_idx}.html")
    fig.write_html(output_html)
    return output_html


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print("1. Loading MONAI Decathlon Dataset...")
    data_dir = os.path.join(script_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        SpatialPadd(keys=["image", "label"], spatial_size=(32, 32, 32)),
        ScaleIntensityRangePercentilesd(keys=["image"], lower=5, upper=95, b_min=0.0, b_max=1.0, clip=True),
        ToTensord(keys=["image", "label"]),
    ])

    val_ds = monai.apps.DecathlonDataset(
        root_dir=data_dir,
        task="Task04_Hippocampus",
        section="validation",
        download=False,
        seed=0,
        val_frac=0.2,
        transform=val_transforms,
        cache_num=0
    )
    # Load 5 samples
    val_loader = DataLoader(val_ds, batch_size=1)

    print("2. Initializing 3D U-Net Model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=3,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)

    model_path = os.path.join(script_dir, "best_hippocampus_unet.pth")
    if not os.path.exists(model_path):
        print(f"Error: Could not find {model_path}.")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print("3. Running AI Inference on 5 Validation Patients...\n")
    print("="*60)
    print(f"{'Patient':<12} | {'Volume (mm³)':<15} | {'Diagnosis (3-Class)':<25}")
    print("="*60)

    generated_files = []
    
    # Iterate over the first 5 samples
    for i, val_data in enumerate(val_loader):
        if i >= 5:
            break
            
        val_inputs = val_data["image"].to(device)
        
        with torch.no_grad():
            val_outputs = sliding_window_inference(val_inputs, (32, 32, 32), 4, model)
            val_preds = torch.argmax(val_outputs, dim=1, keepdim=True)

        pred_volume_array = val_preds[0, 0].cpu().numpy()
        mri_volume_array = val_inputs[0, 0].cpu().numpy()
        
        volume_mm3 = np.sum(pred_volume_array > 0)
        diagnosis, color = classify_alzheimers(volume_mm3)
        
        print(f"Sample #{i+1:<4} | {volume_mm3:<15} | {diagnosis:<25}")
        
        # Generate the 3D visualization showing the MRI slices intersecting the mask
        html_file = create_3d_visualization(
            mri_volume=mri_volume_array, 
            mask_volume=pred_volume_array, 
            volume_mm3=volume_mm3, 
            diagnosis=diagnosis, 
            sample_idx=i+1,
            script_dir=script_dir
        )
        if html_file:
            generated_files.append(html_file)

    print("="*60)
    print("\nSUCCESS! Generated 3D interactive models for all 5 patients.")
    for f in generated_files:
        print(f" -> {os.path.basename(f)}")

if __name__ == "__main__":
    main()
