from ultralytics import YOLO

from padelClipsPackage.FeatureExtraction import process_image

# Load a model
#model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Use the model
model.train(data="/home/juliofgx/PycharmProjects/PadelClips/dataset/padel_pove/2set/2set_2/train/data.yaml", epochs=300, imgsz=1920, device=0, batch=8, workers=8)  # train the model
metrics = model.val()  # evaluate model performance on the validation set
#results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
#path = model.export(format="onnx")  # export the model to ONNX format

# ball train11 1920 players-net train20
#model = YOLO("//runs/detect/train20/weights/best.pt")
#model = YOLO("/home/juliofgx/PycharmProjects/PadelClips/runs/detect/train17/weights/best.onnx")
#path = model.export(format="onnx")
#print(path)
#path = model.export(half=True)
#model = YOLO(path)
# Define path to video file
#source = "/home/juliofgx/PycharmProjects/PadelClips/dataset/padel5/padel5_segment3.mp4"

# Run inference on the source
#results = model(source, stream=False, imgsz=1920, save=True, line_width=4)

#source = "/home/juliofgx/PycharmProjects/PadelClips/dataset/padel5/padel5_segment3.mp4"
