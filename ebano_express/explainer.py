from collections import OrderedDict

import numpy as np
from matplotlib.patches import Rectangle
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

from tensorflow.python import keras
from tensorflow.python.keras import backend as K
from PIL import Image, ImageFilter

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import seaborn as sns
import sys

import pandas as pd


class PerturbationScores:

    col_prediction = "prediction"
    col_prediction_t = "prediction_t"
    col_a_PIR = "a_PIR"
    col_b_PIR = "b_PIR"
    col_PIR = "PIR"
    col_nPIR = "nPIR"
    col_a_PIRP = "a_PIRP"
    col_b_PIRP = "b_PIRP"
    col_PIRP = "PIRP"
    col_nPIRP  = "nPIRP"

    def __init__(self, P_o, P_t, coi):

        self.coi = coi

        self.P_o = P_o
        self.p_o = self.P_o[self.coi]

        if P_t is not None:
            self.P_t = P_t
            self.p_t = self.P_t[self.coi]
        else:
            self.P_t = None
            self.p_t = float('NaN')

        self.PIR = float('NaN')
        self.PIRP = float('NaN')

        self.nPIR = float('NaN')
        self.nPIRP = float('NaN')

        self.a_pir, self.b_pir = float('NaN'), float('NaN')
        self.classes_npir = float('NaN')
        self.w_c_npir = float('NaN')
        self.pirp_coi = float('NaN')
        self.pirp_no_coi = float('NaN')
        self.a_pirp, self.b_pirp = float('NaN'), float('NaN')

    @staticmethod
    def softsign_norm(x):
        x_n = x / (1 + abs(x))
        return x_n

    @staticmethod
    def relu(x):
        if x >= 0:
            return x
        else:
            return 0.0

    @staticmethod
    def _get_a_b(p_o, p_t):

        a = (1 - p_o / p_t)

        if a == float('inf'):
            a = sys.float_info.max
        elif a == float('-inf'):
            a = -sys.float_info.max

        b = (1 - p_t / p_o)

        if b == float('inf'):
            b = sys.float_info.max
        elif b == float('-inf'):
            b = -sys.float_info.max

        return a, b

    @staticmethod
    def compute_influence_relation(p_o, p_t):
        a, b = PerturbationScores._get_a_b(p_o, p_t)
        return (p_t * b) - (p_o * a)

    @staticmethod
    def compute_perturbation_influence_relation(p_o, p_t):
        return PerturbationScores.compute_influence_relation(p_o, p_t)

    @staticmethod
    def compute_perturbation_influence_relation_normalized(p_o, p_t):
        PIR = PerturbationScores.compute_perturbation_influence_relation(p_o, p_t)
        return PerturbationScores.softsign_norm(PIR)

    @staticmethod
    def compute_npir_for_all_classes(P_o, P_t):
        classes_npir = [PerturbationScores.compute_perturbation_influence_relation_normalized(p_o, p_t) for p_o, p_t in zip(P_o, P_t)]
        return classes_npir

    @staticmethod
    def weighted_classes_npir(classes_npir, weights):
        return classes_npir * weights

    @staticmethod
    def pirp_coi(w_c_npir, coi):
        pirp_coi = abs(w_c_npir[coi])
        return pirp_coi

    @staticmethod
    def pirp_no_coi(w_c_npir, coi):
        w_c_npir_no_coi = w_c_npir.copy()
        w_c_npir_no_coi[coi] = 0.0
        w_c_npir_no_coi = [PerturbationScores.relu(wir) for wir in w_c_npir_no_coi]
        pirp_no_coi = sum(w_c_npir_no_coi)
        return pirp_no_coi

    @staticmethod
    def compute_perturbation_influence_relation_precision(P_o, P_t, coi):
        classes_npir = PerturbationScores.compute_npir_for_all_classes(P_o, P_t)

        w_c_npir = PerturbationScores.weighted_classes_npir(classes_npir, P_o)

        pirp_coi = PerturbationScores.pirp_coi(w_c_npir, coi)
        pirp_no_coi = PerturbationScores.pirp_no_coi(w_c_npir, coi)

        return PerturbationScores.compute_influence_relation(pirp_coi, pirp_no_coi)

    @staticmethod
    def compute_perturbation_influence_relation_precision_normalized(P_o, P_t, coi):
        """
        se new_irp_simm > 0 -> la feature è precisa nella la classe in esame \n
        se new_irp_simm = 0 -> la feature non è precisa nella la classe in esame ma impatta anche altre classi \n
        se new_irp_simm < 0 -> la feature non è precisa nella la classe in esame e impatta maggiormente altre classi \n\n

        :param P_o:
        :param P_t:
        :param coi:
        :return:
        """
        pirp = PerturbationScores.compute_perturbation_influence_relation_precision(P_o, P_t, coi)
        return PerturbationScores.softsign_norm(pirp)

    def compute_scores(self):

        if self.P_t is None:
            return self

        self.PIR = PerturbationScores.compute_perturbation_influence_relation(self.p_o, self.p_t)
        self.nPIR = PerturbationScores.compute_perturbation_influence_relation_normalized(self.p_o, self.p_t)

        self.PIRP = PerturbationScores.compute_perturbation_influence_relation_precision(self.P_o, self.P_t, self.coi)
        self.nPIRP = PerturbationScores.compute_perturbation_influence_relation_precision_normalized(self.P_o, self.P_t, self.coi)

        self.a_pir, self.b_pir = PerturbationScores._get_a_b(self.p_o, self.p_t)

        self.classes_npir = PerturbationScores.compute_npir_for_all_classes(self.P_o, self.P_t)

        self.w_c_npir = PerturbationScores.weighted_classes_npir(self.classes_npir, self.P_o)

        self.pirp_coi = PerturbationScores.pirp_coi(self.w_c_npir, self.coi)
        self.pirp_no_coi = PerturbationScores.pirp_no_coi(self.w_c_npir, self.coi)

        self.a_pirp, self.b_pirp = PerturbationScores._get_a_b(self.pirp_coi, self.pirp_no_coi)

        return self

    def get_scores_dict(self):
        scores_dict = OrderedDict()

        scores_dict[self.col_prediction] = float(self.p_o)
        scores_dict[self.col_prediction_t] = float(self.p_t)

        # PIR - Perturbation Influence Relation
        scores_dict[self.col_a_PIR] = float(self.a_pir)
        scores_dict[self.col_b_PIR] = float(self.b_pir)
        scores_dict[self.col_PIR] = float(self.PIR)
        scores_dict[self.col_nPIR] = float(self.nPIR)

        # PIRP - Perturbation Influence Relation Precision
        scores_dict[self.col_a_PIRP] = float(self.a_pirp)
        scores_dict[self.col_b_PIRP] = float(self.b_pirp)
        scores_dict[self.col_PIRP] = float(self.PIRP)
        scores_dict[self.col_nPIRP] = float(self.nPIRP)

        return scores_dict

    def __str__(self):
        return str(self.get_scores_dict())


class LocalExplanation:

    def __init__(self, input_image, class_of_interest, features_map, model, preprocess_func):
        self.input_image = input_image
        self.class_of_interest = class_of_interest
        self.features_map = features_map
        self.model=model
        self.preprocess_func = preprocess_func
        self.cmap = sns.diverging_palette(20, 130, l=60, as_cmap=True)  # 20 -> red, 130 -> green

        self.original_predictions = self.predict_with_model(input_image)
        self.feature_ids = np.unique(self.features_map)

        self.perturbations = {}
        self.numerical_explanation = {}
        self.visual_explanation = None

        self.informativeness = None

    def predict_with_model(self, img):
        if self.preprocess_func:
            input_image_arr = self.preprocess_func(img)
        else:
            raise Exception("Preprocessing function not provided. You should always provide the same preprocess function used on the input image.")

        p = self.model.predict(input_image_arr)
        return p[0]

    @staticmethod
    def _find_centroid(init_x, X, n_iter=10):
        new_x = init_x
        f_X = X

        for i in range(n_iter):
            dist = np.linalg.norm(f_X - new_x, axis=1)
            if f_X[dist < np.mean(dist)].__len__() > 0:
                f_X = f_X[dist < np.mean(dist)]
            else:
                break

            new_x = np.percentile(f_X, 50, interpolation="nearest", axis=0).astype(np.int64)
        return new_x

    @staticmethod
    def _color_palette():
        return (np.array(sns.color_palette("Set3")) * 255).astype(np.uint8)

    @staticmethod
    def _color_rgba(i, alpha=255):
        r, g, b = LocalExplanation._color_palette()[i, :]
        return (r, g, b, alpha)

    def show_features_map(self):

        colored_feature_map = np.zeros((self.features_map.shape[0], self.features_map.shape[0], 4), np.int64)
        coordinates = []

        for f_idx in self.feature_ids:
            feature_map_mask = np.zeros((self.features_map.shape[0], self.features_map.shape[0], 4), np.int64) # init empty mask
            c = self._color_rgba(f_idx - 1) # color

            f = self._get_feature_mask(f_idx)

            x_coors, y_coors = np.where(f == 255)
            y_coor, x_coor = np.percentile(x_coors, 50).astype(int), np.percentile(y_coors, 50).astype(int)

            coors = np.array([x_coors, y_coors]).transpose(1, 0)
            coor = np.array([(x_coor, y_coor)])

            coor = self._find_centroid(coor, coors)
            y_coor, x_coor = coor[0], coor[1]

            # set_color
            feature_map_mask[x_coors, y_coors] = c

            colored_feature_map += feature_map_mask

            coordinates.append((f_idx, x_coor, y_coor, c))

        fig, ax = plt.subplots(figsize=(5, 5), dpi=90)
        fig.tight_layout()
        ax.imshow(colored_feature_map)

        for fid, x_coor, y_coor, c in coordinates:
            ax.text(x_coor, y_coor, f"{fid}", color="black", ha='center', va='center', fontsize=26)

        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Features map")

        return ax

    def show_visual_explanation(self, color_bar=True):
        if self.visual_explanation:
            fig, ax = plt.subplots(figsize=(5, 5))
            fig.tight_layout()
            ax.imshow(self.visual_explanation)
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])

            if color_bar:
                cb_ax = fig.add_axes([1, 0.12, 0.05, 0.8])
                cb = matplotlib.colorbar.ColorbarBase(cb_ax, cmap=self.cmap,
                                       norm=matplotlib.colors.Normalize(vmin=-1, vmax=1),
                                       orientation='vertical')
                cb.set_label(PerturbationScores.col_nPIR)
            return ax
        else:
            raise Exception("Fit explanation first.")

    def show_numerical_explanation(self):

        if self.numerical_explanation:

            _df = pd.DataFrame(self.numerical_explanation).T.sort_index()

            fig, ax = plt.subplots(figsize=(5, 5), dpi=90)
            fig.tight_layout()
            _df[["nPIR", "nPIRP"]].plot(kind="bar", ax=ax, rot=0)
            ax.set_xlabel("Feature id")
            ax.set_title("Class of interest: " + str(self.class_of_interest))
            ax.set_ylim(-1, 1)
            ax.grid(True)

            return ax
        else:
            raise Exception("Fit explanation first.")

    def get_perturbed_input(self, feature_mask):
        im = self.input_image.copy()
        blurred_image = im.filter(ImageFilter.GaussianBlur(10))
        im.paste(blurred_image, mask=feature_mask)
        return im

    def _get_feature_mask(self, f_idx):
        f = self.features_map.reshape(-1).copy()
        f = (f == f_idx).astype(np.uint8) * 255
        f = f.reshape(self.features_map.shape)
        return f

    def get_feature_mask_image(self, f_idx):
        f = self._get_feature_mask(f_idx)
        f_map_img = Image.fromarray(f, mode="L")
        return f_map_img

    def _compute_perturbations(self):

        for f_idx in self.feature_ids:
            feature_mask = self.get_feature_mask_image(f_idx)
            perturbed_image = self.get_perturbed_input(feature_mask)

            self.perturbations[f_idx] = perturbed_image

        return self

    @staticmethod
    def _get_visual_explanation_mask(heatmap, cmap):
        colors = plt.cm.ScalarMappable(cmap=cmap).to_rgba(heatmap)
        mask = Image.fromarray((colors*255).astype(np.uint8), mode="RGBA")

        return mask

    @staticmethod
    def _apply_mask(image, heatmap):
        res = Image.blend(image, heatmap, alpha=.85)
        return res

    def _explain(self):

        # numerical explanation
        for f_idx, img_p in self.perturbations.items():
            predictions_p = self.predict_with_model(img_p)
            perturbation_scores = PerturbationScores(self.original_predictions, predictions_p, self.class_of_interest)
            perturbation_scores.compute_scores()

            self.numerical_explanation[f_idx] = perturbation_scores.get_scores_dict()

        # visual explanation
        heatmap_nPIR = np.zeros(self.features_map.shape, np.float32)  # init empty mask
        for f_idx in self.feature_ids:
            f = self._get_feature_mask(f_idx).astype(np.float32)
            f = f * self.numerical_explanation[f_idx][PerturbationScores.col_nPIR]
            heatmap_nPIR += f

        visual_explanation_mask = self._get_visual_explanation_mask(heatmap_nPIR, cmap=self.cmap)
        self.visual_explanation = self._apply_mask(self.input_image.copy().convert("RGBA"), visual_explanation_mask)

        # Informativeness
        nPIR_values = [scores[PerturbationScores.col_nPIR] for scores in self.numerical_explanation.values()]
        nPIR_values_max = np.max(nPIR_values)
        nPIR_values_min = np.min(nPIR_values)
        self.informativeness = nPIR_values_max - nPIR_values_min

        return self

    def fit_explanation(self):
        return self._compute_perturbations()._explain()

    def get_numerical_explanation(self):
        if self.numerical_explanation:
            _df = pd.DataFrame(self.numerical_explanation).T.sort_index()
            return _df
        else:
            raise Exception("Fit explanation first.")

    def get_visual_explanation(self):
        if self.visual_explanation:
            return self.visual_explanation
        else:
            raise Exception("Fit explanation first.")

    def get_interpretable_feature(self, f_idx):
        if (f_idx < min(self.feature_ids)) or (f_idx > max(self.feature_ids)):
            raise Exception(f"Feature idx '{f_idx}' out of range.")

        feature_mask = self.get_feature_mask_image(f_idx)
        masked_feature = self.input_image.copy()
        masked_feature.putalpha(feature_mask)
        return masked_feature

    def __str__(self):
        return (f"LocalExplanation\
        Class of interest: {self.class_of_interest}\
        # Inf. Feat.: {self.feature_ids.__len__()}\
        {'Informativeness:' + str(round(self.informativeness,2)) if self.informativeness else 'Not fitted'}")


class LocalExplanationModel:

    def __init__(self,  input_image, class_of_interest, model, preprocess_func, max_features=10, k_pca=30, layers_to_analyze=None):
        self.input_image = input_image
        self.class_of_interest = class_of_interest
        self.model = model
        self.model_input_shape = self.model._feed_input_shapes[0][1:3]

        self.k_pca = k_pca
        self.max_features = max_features

        self.preprocess_func = preprocess_func

        # Layers
        self.layer_indexes = self._get_conv_layer_indexes(self.model)
        # print(self.layer_indexes)
        if not layers_to_analyze:
            layers_to_analyze = int(np.log(self.layer_indexes.__len__())/np.log(2))
        else:
            if layers_to_analyze > self.layer_indexes.__len__():
                raise Exception(f"# layers to analyze has to be lower or eqaul to the number of available convolutional layers '{self.layer_indexes.__len__()}'.")

        # print("# layers_to_analyze:", layers_to_analyze)
        self.layer_indexes = self.layer_indexes[-layers_to_analyze:]
        # print(self.layer_indexes)
        self.layers = [self.model.layers[li].output for li in self.layer_indexes]
        self.get_feature_func = K.function([self.model.layers[0].input], self.layers)

        # Explanation
        self.local_explanations = {}
        self.best_explanation = None

    @staticmethod
    def _get_conv_layer_indexes(model):
        layer_indexes = []
        i = 0
        for l in model.layers:
            layer_name = str.lower(l.get_config()["name"])
            if (isinstance(l, keras.layers.Conv2D)) | ("conv2d" in layer_name):
                layer_indexes.append(i)
            i = i + 1
        return layer_indexes

    def get_model_features(self, image_arr):
        return self.get_feature_func([image_arr, 0])

    def get_hypercolumns(self, input_image):
        feature_maps = self.get_model_features(input_image)

        hypercolumns = []
        count = 0
        for convmap in feature_maps:

            reshaped_convmap = convmap[0].transpose((2, 1, 0))

            for fmap in reshaped_convmap:
                upscaled = np.array(Image.fromarray(fmap).resize(self.model_input_shape, Image.BILINEAR))
                hypercolumns.append(upscaled)
            count = count + 1

        hypercolumns = np.asarray(hypercolumns)
        return hypercolumns

    def _features_reduction(self, X):
        pca = PCA(n_components=self.k_pca, copy=False)
        X_r = pca.fit_transform(X)
        return X_r

    def extract_hypercolumns(self, input_image, verbose=False):
        x = self.preprocess_func(input_image)

        hc = self.get_hypercolumns(x)
        # print("Hypecolumns shape:", hc.shape) if verbose else None

        hc_t = hc.transpose([2, 1, 0]).reshape(self.model_input_shape[0] * self.model_input_shape[1], -1)  # e.g. 224*224 = 50176  color transpose, tensor reshape
        # print("Hypecolumns reshaped:", hc_t.shape) if verbose else None

        hc_r = self._features_reduction(hc_t)
        # print("Hypecolumns reduced:", hc_r.shape) if verbose else None

        return hc_r

    def compute_features_map(self, hc, n_features):
        hc_n = normalize(hc, norm='l2', axis=1)
        kmeans_model = KMeans(n_clusters=n_features, max_iter=300, n_jobs=-1)
        features_labels = kmeans_model.fit_predict(hc_n)
        features_map = features_labels.reshape(self.model_input_shape[0], self.model_input_shape[1]).astype(np.uint8)
        features_map += 1
        return features_map

    def fit_explanation(self, verbose=False):

        hc = self.extract_hypercolumns(self.input_image, verbose=verbose)

        for n_f in range(2, self.max_features+1):
            features_map = self.compute_features_map(hc, n_f)
            print(f"> Computing explanation with '{n_f}' features...") if verbose else None

            local_expl_model = LocalExplanation(self.input_image, self.class_of_interest, features_map, self.model,
                                                preprocess_func=self.preprocess_func)

            local_expl_model.fit_explanation()
            self.local_explanations[n_f] = local_expl_model

        self.best_explanation = self.compute_best_explanation()

        return self

    def compute_best_explanation(self):
        if self.local_explanations.items().__len__() == 0:
            raise Exception("Fit explanation first.")

        return max(self.local_explanations.values(), key=lambda x: x.informativeness)


class ClassGlobalExplanationModel:

    def __init__(self, input_images, class_of_interest, model, preprocess_func, max_features=10, k_pca=30, layers_to_analyze=None):
        self.model = model
        self.preprocess_func=preprocess_func
        self.max_features = max_features
        self.k_pca = k_pca
        self.layers_to_analyze = layers_to_analyze

        self.input_images = input_images
        self.class_of_interest = class_of_interest
        self.local_explanation_models = {}
        self.global_explanation = []

    def fit_explanation(self, verbose=False):
        for idx, im in enumerate(self.input_images):
            local_expl_model = LocalExplanationModel(im, self.class_of_interest, self.model, self.preprocess_func,
                                                     max_features=self.max_features, k_pca=self.k_pca,
                                                     layers_to_analyze=self.layers_to_analyze)

            local_expl_model = local_expl_model.fit_explanation()
            self.local_explanation_models[idx] = local_expl_model.best_explanation

        for im_idx, l_expl in self.local_explanation_models.items():
            for f_idx in l_expl.feature_ids:
                f_im = l_expl.get_interpretable_feature(f_idx)
                f_score = l_expl.get_numerical_explanation().loc[f_idx, [PerturbationScores.col_nPIR, PerturbationScores.col_nPIRP]].to_dict()
                self.global_explanation.append({"image_id":im_idx, "feature_id":f_idx, "feature_img": f_im, "feature_scores": f_score})

        return self

    @staticmethod
    def _imscatter(x, y, image, ax=None, zoom=1):
        if ax is None:
            ax = plt.gca()

        im = OffsetImage(image, zoom=zoom)
        x, y = np.atleast_1d(x, y)
        artists = []
        for x0, y0 in zip(x, y):
            ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=True)
            artists.append(ax.add_artist(ab))
            ax.scatter(x0, y0)
        ax.update_datalim(np.column_stack([x, y]))
        ax.autoscale()
        return artists

    def show_global_explanation(self):
        if self.global_explanation.__len__() == 0:
            raise Exception("Fit explanation first.")

        t_size = (80, 80)

        f_p_report = self.global_explanation.copy()
        f_p_report = sorted(f_p_report, key=lambda x: np.abs(x["feature_scores"][PerturbationScores.col_nPIR]))

        # KDE plot
        lm = sns.jointplot([fp["feature_scores"][PerturbationScores.col_nPIR] for fp in f_p_report],
                           [fp["feature_scores"][PerturbationScores.col_nPIRP] for fp in f_p_report],
                           kind="kde", height=8)
        ax = lm.ax_joint
        ax.clear()

        # Image plot
        for g_expl in f_p_report:
            f_im = g_expl["feature_img"]
            f_score = g_expl["feature_scores"]

            t_im = f_im.copy()
            t_im = t_im.resize(t_size, Image.ANTIALIAS)
            t_im = np.array(t_im)

            self._imscatter(f_score[PerturbationScores.col_nPIR], f_score[PerturbationScores.col_nPIRP], t_im, ax=ax, zoom=1)

        ax.set_ylim(-1.25, 1.25)
        ax.set_xlim(-1.15, 1.15)
        ax.set_xlabel("nPIR")
        ax.set_ylabel("nPIRP")
        lm.ax_marg_x.set_title("Class of interest:" + str(self.class_of_interest))

        ax.axvline(0.0, 0.0, c='black')
        ax.axhline(0.0, 0.0, c='black')

        # Create a Rectangle patch
        rect1 = Rectangle((-1.15, -1.25), 0.15, 2.5, linewidth=0, edgecolor='r', facecolor='gray', alpha=0.5)
        rect2 = Rectangle((1.0, -1.25), 0.15, 2.5, linewidth=0, edgecolor='r', facecolor='gray', alpha=0.5)

        rect3 = Rectangle((-1.0, 1.0), 2.0, 0.25, linewidth=0, facecolor='gray', alpha=0.5)
        rect4 = Rectangle((-1.0, -1.25), 2.0, 0.25, linewidth=0, facecolor='gray', alpha=0.5)

        # Add the patch to the Axes
        ax.add_patch(rect1)
        ax.add_patch(rect2)
        ax.add_patch(rect3)
        ax.add_patch(rect4)

        ax.grid(True)

        return ax, lm





