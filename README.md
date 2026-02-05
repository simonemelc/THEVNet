# Fusing Thermal and Event Data for Visible Spectrum Image Reconstruction

[![Conference](https://img.shields.io/badge/Conference-VISAPP_2026-blue)](https://visapp.scitevents.org/)

**Authors:** Simone Melcarne and Jean-Luc Dugelay

*Eurecom Research Center, Digital Security Department, Biot, France*

---

<p align="center">
  <img src="figures/visual_comparison.PNG" alt="Visual" width="100%">
</p>

## Publication
This paper has been accepted for publication at the **21st International Conference on Computer Vision Theory and Applications (VISAPP 2026)**.

---

## Abstract
Reconstructing visible spectrum images from unconventional sensors is a timely and relevant problem in computer vision. In settings where standard cameras fail or are not allowed, thermal and event-based cameras can offer complementary advantages—robustness to darkness, fog, motion, and high dynamic range conditions—while also being privacy-preserving and energy efficient.

We propose a simple **dual-encoder, gated-fusion network** that synthesizes visible-like images from thermal frames and event streams. The thermal branch captures structure and coarse appearance, while the event branch models spatio-temporal changes and adds detailed edge information. Their outputs are combined via a residual gated fusion mechanism and decoded into a colored image.

## The Framework

<p align="center">
  <img src="figures/pipeline_fusenet.png" alt="Framework Overview" width="100%">
</p>

Our method leverages the complementary nature of two modalities to reconstruct a standard RGB-like image ($I \in \mathbb{R}^{3 \times H \times W}$):

1.  **Thermal Input ($T$):** Provides semantic layout and object presence (processed by a ResNet-style encoder).
2.  **Event Input ($E$):** A voxel grid of events that offers sharp edge and motion cues (processed by a 3D-Conv head + ResNet encoder).

### Key Components:
* **Dual-Fusion Module:** Features from both encoders are fused at multiple resolutions using a **Gated Fusion Block (GFB)**. This allows the network to selectively inject event details into the thermal baseline:
    $$f_{TE}^{(i)} = f_T^{(i)} + \alpha_i \cdot f_E^{(i)}$$
* **Color Space:** The network predicts in the **Lab color space** to better separate luminance and chrominance.
* **Loss Function:** We train end-to-end using a combination of Spatial L1 ($L_{s-l_1}$), Multi-Scale SSIM ($L_{MS-SSIM}$), and Color L1 ($L_{ab-l_1}$) losses.

## Dataset
We evaluated our approach on the **KAIST-MS dataset** (Daylight set).
* **Thermal & RGB:** Real-world recordings from KAIST-MS.
* **Events:** Synthetically generated from RGB frames using the **V2E simulator** to ensure perfect pixel-alignment.

## Usage

```bash
# Clone the repository
git clone [https://github.com/yourusername/thermal-event-fusion.git](https://github.com/yourusername/thermal-event-fusion.git)

# Install dependencies
pip install -r requirements.txt

# Run inference on a sample
python test.py --thermal input/thermal.png --events input/events.npz --output result.png
```

## Citation

If you find this work useful for your research, please cite our paper:

```bibtex
@inproceedings{melcarne2026fusing,
  title={Fusing Thermal and Event Data for Visible Spectrum Image Reconstruction},
  author={Melcarne, Simone and Dugelay, Jean-Luc},
  booktitle={Proceedings of the 21st International Conference on Computer Vision Theory and Applications (VISAPP)},
  year={2026},
  organization={SciTePress}
}
