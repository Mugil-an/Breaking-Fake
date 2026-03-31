# # import torch
# # import timm
# # from torchvision import transforms
# # from PIL import Image
# # import os

# # # 1. Setup
# # device = torch.device("cpu") # Use CPU for local testing
# # model_path = "breaking_fake_vit.pth" # Ensure this file is in D:\DL

# # # 2. Rebuild the Model Structure
# # model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=2)
# # model.load_state_dict(torch.load(model_path, map_location=device))
# # model.eval()
# # print("✅ Model Loaded Successfully on CPU!")

# # # 3. Simple Prediction Function
# # def predict(path):
# #     img = Image.open(path).convert("RGB")
# #     transform = transforms.Compose([
# #         transforms.Resize((224, 224)),
# #         transforms.ToTensor(),
# #         transforms.Normalize([0.5]*3, [0.5]*3)
# #     ])
# #     img_t = transform(img).unsqueeze(0)
    
# #     with torch.no_grad():
# #         output = model(img_t)
# #         prediction = torch.argmax(output, dim=1).item()
    
# #     # REMEMBER: 0 = AI, 1 = Real
# #     return "AI GENERATED" if prediction == 0 else "REAL PHOTOGRAPH"

# # # 4. RUN THE TEST
# # # Put any image path here (Real or AI)
# # test_image_path = "D:/DL/test1.jpg" 
# # if os.path.exists(test_image_path):
# #     result = predict(test_image_path)
# #     print(f"🔍 TEST RESULT: {result}")
# # else:
# #     print("❌ Please put an image in D:/DL/test_samples/ to test!")
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# def forensic_fft(path):
#     img = cv2.imread(path, 0)
#     img = cv2.resize(img, (224, 224))
    
#     # Transform to Frequency Domain
#     f = np.fft.fft2(img)
#     fshift = np.fft.fftshift(f)
#     magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    
#     plt.figure(figsize=(10, 5))
#     plt.subplot(121), plt.imshow(cv2.imread(path)[:,:,::-1]), plt.title('Original')
#     plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray'), plt.title('FFT Spectrum')
#     plt.show()
    
#     # Calculate Variance: AI images often have specific frequency spikes
#     print(f"Frequency Variance: {np.var(magnitude_spectrum)}")

# forensic_fft("D:/DL/test1.jpg")
print(train_ds.class_to_idx)