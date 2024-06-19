from ultralytics import YOLO

# Load a model
#model = YOLO("yolov8n.yaml")  # build a new model from scratch
#model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Use the model
#model.train(data="./dataset/padel5/train_2_net/data.yaml", epochs=300, imgsz=1920, device=0, batch=8, workers=8)  # train the model
#metrics = model.val()  # evaluate model performance on the validation set
#results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
#path = model.export(format="onnx")  # export the model to ONNX format

# train11 1920 train17 1080
model = YOLO("/home/juliofgx/PycharmProjects/padelLynx/runs/detect/train20/weights/best.pt")
#model = YOLO("/home/juliofgx/PycharmProjects/padelLynx/runs/detect/train17/weights/best.onnx")
#path = model.export(format="onnx")
#print(path)
#path = model.export(half=True)
#model = YOLO(path)
# Define path to video file
source = "/home/juliofgx/PycharmProjects/padelLynx/dataset/padel5/padel5_segment3.mp4"

# Run inference on the source
#results = model(source, stream=False, imgsz=1920, save=True, line_width=4)

#source = "/home/juliofgx/PycharmProjects/padelLynx/dataset/padel5/padel5_segment3.mp4"
# Run inference on the source
results = model(source, stream=True, half=False,imgsz=1920, save=False, save_frames=False, show_conf=True, verbose=False, show_labels=True, line_width=4, save_txt = True, save_conf = True)
i = 0
for r in results:
    next(results)
    if i%100 == 0:
        print(str(i), end='\r', flush=True)
    i += 1