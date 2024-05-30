from ultralytics import YOLO

# Load a model
#model = YOLO("yolov8n.yaml")  # build a new model from scratch
#model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Use the model
#model.train(data="./dataset/padel3/images_and_labels/data.yaml", epochs=300, imgsz=1920, device=0, batch=8)  # train the model
#metrics = model.val()  # evaluate model performance on the validation set
#results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
#path = model.export(format="onnx")  # export the model to ONNX format

model = YOLO("/home/juliofgx/PycharmProjects/padelLynx/runs/detect/train8/weights/best.onnx")
# Define path to video file
#source = "/home/juliofgx/PycharmProjects/padelLynx/dataset/padel3/padel3_segment_3.mp4"

# Run inference on the source
#results = model(source, stream=False, imgsz=1920, save=True, line_width=4)

source = "/home/juliofgx/PycharmProjects/padelLynx/dataset/seg1/segment.mp4"

# Run inference on the source
results = model(source, stream=True, imgsz=1920, save=True, save_frames=True, show_conf=False, show_labels=False, line_width=0, save_txt = True, save_conf = True, vid_stride = 10)
for r in results:
    next(results)