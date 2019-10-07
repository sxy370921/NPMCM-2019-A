import numpy as np
import pandas as pd
from feature.extract_feature2 import extract_feature
from model_service.tfserving_model_service import TfServingBaseService


class mnist_service(TfServingBaseService):

    def __init__(self, model_name, model_path):
        super().__init__(model_name, model_path)
        # self.label_max = -59.64000
        # self.label_min = -123.9400

    def _preprocess(self, data):

        preprocessed_data = {}
        filesDatas = []
        for k, v in data.items():
            for file_name, file_content in v.items():
                pb_data = pd.read_csv(file_content)
                feat = extract_feature(pb_data)
                # input_data = np.array(pb_data.get_values()[:,0:17], dtype=np.float32)
                print(file_name, feat.shape)
                filesDatas.append(feat)

        # filesDatas = np.array(filesDatas,dtype=np.float32).reshape(-1, 17)
        filesDatas = np.vstack(filesDatas).astype('float32')
        preprocessed_data['myInput'] = filesDatas
        print("preprocessed_data[\'myInput\'].shape = ", preprocessed_data['myInput'].shape)

        return preprocessed_data


    def _postprocess(self, data):
        infer_output = {"RSRP": []}
        for output_name, results in data.items():
            print(output_name, np.array(results).shape)
            print(type(results))
            # results = np.array(results)
            # results = results * (self.label_max-self.label_min) + self.label_min
            infer_output["RSRP"] = results
        return infer_output