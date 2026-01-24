import cv2
import torchvision
import torch
import torch.nn as nn
from PIL import Image
from model import myModel

model = myModel(in_f=1, hid_f=64, out_f=7)
model.load_state_dict(torch.load("models/model4(best).pth"))

data_transformer = torchvision.transforms.Compose([
    torchvision.transforms.Grayscale(),
    torchvision.transforms.Resize((48,48)),
    torchvision.transforms.ToTensor()
])
classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    if not success:
        break
    if cv2.waitKey(1) & 0xFF == 27:
        break
    
    PIL_img = Image.fromarray(img)
    img_tensor = data_transformer(PIL_img)
    model.eval()
    with torch.inference_mode():
        img_tensor = img_tensor.unsqueeze(0)
        pred = model(img_tensor)
        softmax = nn.Softmax(dim=1)
        pred_probs = softmax(pred)
        pred_labels = torch.argmax(pred_probs, dim=1)
    
    cv2.putText(img, classes[pred_labels], (100,100), 3, 1, (255,255,255), 1)
    cv2.imshow("Emotion", img)
cap.release()
cv2.destroyAllWindows()