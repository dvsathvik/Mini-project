import os
import torch
import nibabel as nib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet

def classify_alzheimers_from_ratio(ratio):
    # Hippocampal Fraction Ratio = (Volume / eTIV) * 1000
    # For easier reading, the ratio is presented e.g., 6.18, 4.80
    if ratio >= 5.5:
        return "Normal (No AD)", "green"
    elif ratio >= 5.0:
        return "Early AD (Mild Atrophy)", "yellow"
    else:
        return "Late AD (Severe Atrophy)", "red"

def map_cdr_to_class(cdr):
    try:
        cdr_val = float(cdr)
        if cdr_val == 0.0:
            return "Normal (No AD)", "green"
        elif cdr_val == 0.5:
            return "Early AD", "yellow"
        elif cdr_val >= 1.0:
            return "Late AD", "red"
        else:
            return "Unknown", "grey"
    except:
        return "Unknown", "grey"

def create_glass_brain_visualization(full_mri, mask_volume, crop_coords, volume_mm3, ratio, pred_diag, pred_color, true_diag, patient_id, output_path):
    # full_mri: 176x208x176
    # mask_volume: 76x50x40 (the crop)
    # crop_coords: (X_start, Y_start, Z_start)
    
    X_s, Y_s, Z_s = crop_coords
    
    # 1. Smooth Hippocampus 3D Mesh (Isosurface instead of Scatter dots)
    # We apply a slight Gaussian blur to the binary mask so the resulting Isosurface is organically smooth and round, not blocky.
    mask_smooth = gaussian_filter(mask_volume.astype(float), sigma=0.8)
    
    X_m, Y_m, Z_m = np.mgrid[X_s : X_s + mask_volume.shape[0], 
                             Y_s : Y_s + mask_volume.shape[1], 
                             Z_s : Z_s + mask_volume.shape[2]]
                             
    # We map the predicted class color to a Plotly colorscale so the whole organ is one solid color
    color_scale = [[0, pred_color], [1, pred_color]]
    
    fig = go.Figure()
    fig.add_trace(go.Isosurface(
        x=X_m.flatten(),
        y=Y_m.flatten(),
        z=Z_m.flatten(),
        value=mask_smooth.flatten(),
        isomin=0.4,
        isomax=1.0,
        surface_count=1,
        colorscale=color_scale,
        showscale=False,
        opacity=1.0, # Solid
        name="Hippocampus 3D Mesh"
    ))

    # 2. Intersecting Full-Brain MRI Planes (Multiplanar Reconstruction)
    X_shape, Y_shape, Z_shape = full_mri.shape
    
    # Find the center of the crop to place the slices perfectly through the hippocampus
    center_x = X_s + mask_volume.shape[0] // 2
    center_y = Y_s + mask_volume.shape[1] // 2
    center_z = Z_s + mask_volume.shape[2] // 2
    
    # A. AXIAL SLICE (Z is constant)
    x_1d_ax = np.arange(X_shape)
    y_1d_ax = np.arange(Y_shape)
    x_2d_ax, y_2d_ax = np.meshgrid(x_1d_ax, y_1d_ax) # shape: (Y, X)
    z_2d_ax = np.full_like(x_2d_ax, center_z)
    
    # Extract slice, transpose, and make background transparent (nan) to remove ugly grey squares
    color_2d_ax = full_mri[:, :, center_z].astype(float).T 
    color_2d_ax[color_2d_ax < 1e-3] = np.nan
    
    fig.add_trace(go.Surface(
        x=x_2d_ax, y=y_2d_ax, z=z_2d_ax,
        surfacecolor=color_2d_ax,
        colorscale='Greys',
        showscale=False,
        name="Axial Plane",
        opacity=0.9
    ))
    
    # B. CORONAL SLICE (Y is constant)
    x_1d_cor = np.arange(X_shape)
    z_1d_cor = np.arange(Z_shape)
    x_2d_cor, z_2d_cor = np.meshgrid(x_1d_cor, z_1d_cor) # shape: (Z, X)
    y_2d_cor = np.full_like(x_2d_cor, center_y)
    
    color_2d_cor = full_mri[:, center_y, :].astype(float).T
    color_2d_cor[color_2d_cor < 1e-3] = np.nan
    
    fig.add_trace(go.Surface(
        x=x_2d_cor, y=y_2d_cor, z=z_2d_cor,
        surfacecolor=color_2d_cor,
        colorscale='Greys',
        showscale=False,
        name="Coronal Plane",
        opacity=0.9
    ))
    
    # C. SAGITTAL SLICE (X is constant)
    y_1d_sag = np.arange(Y_shape)
    z_1d_sag = np.arange(Z_shape)
    y_2d_sag, z_2d_sag = np.meshgrid(y_1d_sag, z_1d_sag) # shape: (Z, Y)
    x_2d_sag = np.full_like(y_2d_sag, center_x)
    
    color_2d_sag = full_mri[center_x, :, :].astype(float).T
    color_2d_sag[color_2d_sag < 1e-3] = np.nan
    
    fig.add_trace(go.Surface(
        x=x_2d_sag, y=y_2d_sag, z=z_2d_sag,
        surfacecolor=color_2d_sag,
        colorscale='Greys',
        showscale=False,
        name="Sagittal Plane",
        opacity=0.9
    ))
    


    clean_axis = dict(
        showbackground=False,
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        title='',
        showspikes=False,
        showline=False
    )

    fig.update_layout(
        title=f"Patient {patient_id} | Vol: {volume_mm3} mm³ | Ratio: {ratio:.2f}<br>Prediction: {pred_diag} | Ground Truth: {true_diag}",
        scene=dict(
            aspectmode='data',
            xaxis=dict(range=[0, X_shape], **clean_axis), # Force axes to full brain dimensions
            yaxis=dict(range=[0, Y_shape], **clean_axis),
            zaxis=dict(range=[0, Z_shape], **clean_axis),
            bgcolor='rgba(0,0,0,1)' # Pure black background
        ),
        margin=dict(l=0, r=0, b=0, t=60),
        paper_bgcolor='rgba(0,0,0,1)',
        plot_bgcolor='rgba(0,0,0,1)',
        font=dict(color='white')
    )
    
    fig.write_html(output_path)
    return output_path

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    # 1. Load CSV Ground Truth
    csv_path = os.path.join(project_dir, "oasis_cross-sectional.csv")
    df = pd.read_csv(csv_path)
    
    # 2. Init Model
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
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # The 5 OASIS files
    target_files = [
        "OAS1_0001_MR1_mpr_n4_anon_111_t88_masked_gfc.nii",
        "OAS1_0014_MR1_mpr_n4_anon_111_t88_masked_gfc.nii",
        "OAS1_0027_MR1_mpr_n4_anon_111_t88_masked_gfc.nii",
        "OAS1_0308_MR1_mpr_n4_anon_111_t88_masked_gfc.nii",
        "OAS1_0351_MR1_mpr_n4_anon_111_t88_masked_gfc.nii"
    ]
    
    print("="*110)
    print(f"{'Patient ID':<15} | {'Vol (mm³)':<10} | {'eTIV (cm³)':<10} | {'Ratio':<6} | {'Prediction':<25} | {'Ground Truth':<20}")
    print("="*110)
    
    for fname in target_files:
        patient_id = fname.split("_mpr")[0]
        
        row = df[df['ID'] == patient_id]
        if len(row) > 0:
            cdr = row.iloc[0]['CDR']
            etiv = row.iloc[0]['eTIV']
            true_diag, true_color = map_cdr_to_class(cdr)
        else:
            true_diag, true_color = "Unknown", "grey"
            etiv = 1500 # Fallback
            
        nii_path = os.path.join(project_dir, fname)
        if not os.path.exists(nii_path):
            continue
            
        img = nib.load(nii_path).get_fdata()
        img = np.squeeze(img)
        
        # Hardcoded ROI Crop
        X_s, Y_s, Z_s = 50, 90, 50
        img_cropped = img[X_s:X_s+76, Y_s:Y_s+50, Z_s:Z_s+40]
        
        # Preprocess
        img_tensor = torch.tensor(img_cropped, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        lower = np.percentile(img_cropped, 5)
        upper = np.percentile(img_cropped, 95)
        img_tensor = torch.clamp((img_tensor - lower) / (upper - lower + 1e-8), 0.0, 1.0)
        
        with torch.no_grad():
            val_outputs = sliding_window_inference(img_tensor, (32, 32, 32), 4, model)
            val_preds = torch.argmax(val_outputs, dim=1, keepdim=True)
            
        mask = val_preds[0, 0].cpu().numpy()
        volume = np.sum(mask > 0)
        
        # Normalized Ratio
        try:
            ratio = (volume / float(etiv))
        except:
            ratio = 0.0
            
        pred_diag, pred_color = classify_alzheimers_from_ratio(ratio)
        
        print(f"{patient_id:<15} | {volume:<10} | {etiv:<10} | {ratio:<6.2f} | {pred_diag:<25} | {true_diag:<20}")
        
        out_html = os.path.join(script_dir, f"OASIS_IntersectingPlanes_{patient_id}.html")
        create_glass_brain_visualization(
            full_mri=img, 
            mask_volume=mask, 
            crop_coords=(X_s, Y_s, Z_s), 
            volume_mm3=volume, 
            ratio=ratio,
            pred_diag=pred_diag, 
            pred_color=pred_color, 
            true_diag=true_diag, 
            patient_id=patient_id, 
            output_path=out_html
        )

if __name__ == '__main__':
    main()
