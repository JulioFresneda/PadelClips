import os.path

from ultralytics import YOLO
from padelClipsPackage.FeatureExtraction import process_image, FeatureExtractor

import pandas as pd
import numpy as np

class Inference:
    def __init__(self, ball_model_path, players_model_path):
        self.ball_model = YOLO(ball_model_path)
        self.players_model = YOLO(players_model_path)

    def inference(self, source, output_folder, conf):
        self.conf = conf
        ball_model = self.ball_model.track(source, stream=True, half=False, imgsz=1920, save=True, save_frames=False, show_conf=True,
                        verbose=True, show_labels=True, line_width=4, save_txt=False, save_conf=False)

        player_model = self.players_model.track(source, stream=True, half=False, imgsz=1920, save=False, save_frames=False,
                                     show_conf=True,
                                     verbose=False, show_labels=True, line_width=4, save_txt=False, save_conf=False)

        print("Inferencing players and net...")
        #players_df, players_ft = self.inference_to_df(player_model, add_features=True, track=True)
        #players_df.to_excel(os.path.join(output_folder, "players_inference.xlsx"), index=False)
        #players_ft = {f'{tag}': features for tag, features in zip(players_ft['tags'], players_ft['features'])}
        #np.savez_compressed(os.path.join(output_folder, "players_inference_features.npz"), **players_ft)

        print("Inferencing ball...")
        ball_df, _ = self.inference_to_df(ball_model, track=False)
        ball_df.to_excel(os.path.join(output_folder, "ball_inference.xlsx"), index=False)




    def add_features(self, df, image):
        feature_extractor = FeatureExtractor()

        def apply_features(row):
            # Extracts features based on bounding box coordinates and image
            x, y, w, h = row['xywh']
            x, y, w, h = int(x.cpu().item()), int(y.cpu().item()), int(w.cpu().item()), int(h.cpu().item())
            x, y = x - w // 2, y - h // 2
            image_portion = image[y:y + h, x:x + w]
            features = feature_extractor.extract_deep_features(image_portion)
            tag = row['id']
            return pd.Series([features, tag])

        # Group by 'id' and find the row with the highest confidence for each group
        idx = df.groupby('id')['conf'].idxmax()
        result = df.loc[idx].apply(apply_features, axis=1)
        result.columns = ['features', 'tags']

        # Return two separate Series or arrays as required
        return result





    def inference_to_df(self, model, add_features = False, track = False):



        results_df = {'frame':[], 'class':[], 'x':[], 'y':[], 'w':[], 'h':[], 'conf':[], 'id':[], 'xywh':[]}



        image = None
        for result in model:

            boxes = result.boxes  # Boxes object for bounding box outputs
            conf = boxes.conf
            xywhn = boxes.xywhn
            xywh = boxes.xywh
            cls = boxes.cls
            if track:
                id = boxes.id
            else:
                id = pd.Series([None] * len(cls))
            image = result.orig_img



            for (nx, ny, nw, nh), id, c, cl, xywh_t in zip(xywhn, id, conf, cls, xywh):
                if c.cpu().item() >= self.conf:
                    fn = model.gi_frame.f_locals['gen'].gi_frame.f_locals['self'].seen -1

                    results_df['frame'].append(fn)
                    results_df['class'].append(int(cl.cpu().item()))

                    results_df['x'].append(nx.cpu().item())
                    results_df['y'].append(ny.cpu().item())
                    results_df['w'].append(nw.cpu().item())
                    results_df['h'].append(nh.cpu().item())

                    results_df['conf'].append(c.cpu().item())

                    if add_features:
                        results_df['xywh'].append(xywh_t)
                    else:
                        results_df['xywh'].append(None)

                    if track:
                        results_df['id'].append(int(id.cpu().item()))
                    else:
                        results_df['id'].append(None)



            if fn % 100 == 0:
                print("Inferenced frame " + str(fn), end='\r', flush=True)

        results_df = pd.DataFrame(results_df)
        if add_features:
            features = self.add_features(results_df, image)
        else:
            features = None

        results_df = results_df.drop('xywh', axis=1)

        return results_df, features


model_ball = "/home/juliofgx/PycharmProjects/PadelClipsTraining/runs/detect/train5/weights/best.pt"
model_players = "/home/juliofgx/PycharmProjects/PadelClips/models/players/weights/best.pt"
inference = Inference(model_ball, model_players)

source = "/home/juliofgx/PycharmProjects/PadelClips/dataset/padel_pove/1set/1set_1/2set_1.mp4"
inference.inference(source, "/home/juliofgx/PycharmProjects/PadelClips/dataset/padel_pove/2set/2set_1", conf=0.25)

