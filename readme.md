3D Medical Image Segmentation

This project is about building a Python workflow for 3D medical image segmentation using MONAI and PyTorch.

Dataset: Medical Segmentation Decathlon, spleen segmentation task

The project includes the main steps of a medical image segmentation workflow:

- loading CT images and segmentation masks
- preprocessing the 3D medical images
- applying basic augmentation
- training a 3D U-Net model
- validating the model with Dice score
- saving the trained model
- visualizing


python data-preparation.py
python train_model.py --epochs 50
python evaluate_model.py --limit 5