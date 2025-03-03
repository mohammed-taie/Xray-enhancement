<<<<<<< HEAD
# Xray-enhancement
Implement and analyze the impact of ten different image enhancement techniques on medical images, particularly chest X-ray (CXR) images
=======
# Medical X-ray Image Enhancement Techniques

## **Objective**
The goal of this project is to implement and analyze the impact of ten different image enhancement techniques on medical images, specifically chest X-ray (CXR) images. These techniques aim to optimize feature extraction for deep learning-based classification models by improving image quality.

## **Implemented Image Enhancement Techniques**
The code applies the following ten enhancement methods:

1. **Histogram Equalization (HE)** – Improves global contrast by redistributing pixel intensity values.
2. **Contrast Limited Adaptive Histogram Equalization (CLAHE)** – Enhances local contrast while preventing over-amplification of noise.
3. **Image Complement** – Inverts pixel values to highlight inverse features for better recognition.
4. **Gamma Correction** – Adjusts mid-range pixel values to emphasize critical details.
5. **Balance Contrast Enhancement Technique (BCET)** – Dynamically balances contrast without distorting the histogram.
6. **Adaptive Gamma Correction** – Applies gamma correction adaptively for enhanced local contrast.
7. **Laplacian Filtering** – Enhances edge details for better structural visualization.
8. **Unsharp Masking** – Sharpens images by amplifying high-frequency components.
9. **Wavelet Transform** – Enhances contrast while reducing noise.
10. **Top-Hat & Bottom-Hat Filtering** – Extracts small bright and dark structures that may be obscured.

## **Motivation and Importance**
Medical X-ray images often suffer from noise, poor contrast, and varying illumination conditions, which can affect the accuracy of automated analysis systems. The primary motivation behind implementing these techniques is to:
- Improve the **visual quality** of X-ray images.
- Enhance **contrast, edge sharpness, and feature visibility**.
- Aid **machine learning and deep learning models** in making more accurate disease predictions.
- Assist in the **detection of medical conditions** such as tuberculosis and other lung diseases.

## **Analysis of Enhancement Techniques**
Each enhancement technique is applied to a sample X-ray image, and its impact is visualized. The following key benefits are observed:
- **Histogram Equalization (HE) and CLAHE** effectively improve global and local contrast.
- **Image Complement** reveals inverse patterns that may be diagnostically relevant.
- **Gamma Correction and Adaptive Gamma Correction** enhance mid-tone details for better feature extraction.
- **BCET** optimally balances contrast while preserving histogram distribution.
- **Laplacian Filtering and Unsharp Masking** improve edge details for better medical interpretation.
- **Wavelet Transform** aids in noise reduction and contrast enhancement simultaneously.
- **Top-Hat & Bottom-Hat Filtering** highlight small bright and dark features that may be crucial for diagnosis.

## **Conclusion**
By implementing these image enhancement techniques, medical images are **better prepared for further analysis**, ensuring that deep learning models receive high-quality inputs. This enhancement plays a crucial role in **improving disease detection performance**, enabling AI-driven diagnostic systems to provide more accurate results.

>>>>>>> 07d7cf6 (Initial commit - X-ray enhancement app)
