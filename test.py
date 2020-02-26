import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from imblearn.metrics import classification_report_imbalanced
import pandas as pd
from net_model import *
from sklearn.utils import class_weight
import seaborn as sns
from tqdm import tqdm
# 设置cpu运行程序
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.size'] = 6
sns.set(font='SimHei')


def heatmap(df, row_labels, col_labels, ax=None,
            cbar_kw={}, **kwargs):
    if not ax:
        ax = plt.gca()
    im = ax.imshow(df, **kwargs)
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    ax.set_xticks(np.arange(df.shape[1]))
    ax.set_yticks(np.arange(df.shape[0]))
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)
    plt.setp(ax.get_xticklabels(), ha="right",
             rotation_mode="anchor")
    for edge, spine in ax.spines.items():
        spine.set_visible(False)
    ax.set_xticks(np.arange(df.shape[1]), minor=True)
    ax.set_yticks(np.arange(df.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    return im, cbar


def annotate_heatmap(im, df=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    if not isinstance(df, (list, np.ndarray)):
        df = im.get_array()
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(df.max()) / 2.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)
    texts = []
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            kw.update(color=textcolors[int(im.norm(df[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(df[i, j], None), **kw)
            texts.append(text)
    return texts



path = "./Weather_DataSet"
train_images = []
train_labels = []
test_images = []
test_labels = []
width = 75
height = 75
for root, dirs, names in tqdm(os.walk(path, topdown=True)):
    for filename in names:
        file_path = os.path.join(root, filename)
        if "Test" in file_path:
            label = file_path.split("\\")[-2][5:].lower()
        else:
            label = file_path.split("\\")[-2].lower()
        file = cv2.imread(file_path)
        im = cv2.resize(file, (width, height))
        if "Test" in file_path:
            test_images.append(im.reshape(1, width, height, 3) / 255.0)
            test_labels.append(label)
        else:
            train_images.append(im.reshape(1, width, height, 3) / 255.0)
            train_labels.append(label)

        #  flip
        for i in [-1, 0, 1]:
            file_new = cv2.flip(file, i)
            im = cv2.resize(file, (width, height))
            train_images.append(im.reshape(1, width, height, 3) / 255.0)
            train_labels.append(label)


train_images = np.vstack(train_images)
test_images = np.vstack(test_images)
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

class_name = ['Cloudy_bright', 'Cloudy_dark', 'Foggy_bright',
              'Foggy_dark', 'Rainy_bright', 'Rainy_dark',
              'Snowy_bright', 'Snowy_dark', 'Sunny_bright', 'Sunny_dark']

class_name = [x.lower() for x in class_name]
le = LabelEncoder()
le.fit(class_name)
train_labels = le.transform(train_labels)
test_labels = le.transform(test_labels)
pd.value_counts(le.inverse_transform(train_labels))
clean_result()
##
tensorboard_callback = tf.keras.callbacks.TensorBoard()
##提前结束
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=200, restore_best_weights=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, verbose=0, mode='auto',
                                                 min_lr=0.0001)
weight = class_weight.compute_class_weight('balanced', np.unique(train_labels), train_labels)
#
# acc = []
# num_list = [16, 32, 64, 128]
# for num in num_list:
#     print(num)
#     model = SeNet(width, height, num)
#     model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999),
#                   loss='sparse_categorical_crossentropy',
#                   metrics=['accuracy'])
#     history = model.fit(train_images,
#                         train_labels,
#                         epochs=200,
#                         class_weight=dict(enumerate(weight)),
#                         batch_size=128,
#                         verbose=1,
#                         # validation_split=0.3,
#                         validation_data=(test_images, test_labels),
#                         callbacks=[early_stopping,
#                                    reduce_lr])
#     acc.append(model.evaluate(test_images, test_labels)[1])
# plt.figure()
# plt.plot(num_list, acc)
# plt.xlabel('Num')
# plt.ylabel('Accuracy')
# plt.savefig("./result/senet_num.png")
# plt.close()
# pd.DataFrame(acc).to_csv("test.csv", index=False)

result_loss = []
result_predict = []
for model, m_name in zip([google_net(width, height), AlexNet(width, height), SeNet(width, height), BP(width, height)],
                         ["google_net", "alexnet", "SeNet", "BP"]):
    model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(train_images,
                        train_labels,
                        epochs=200,
                        class_weight=dict(enumerate(weight)),
                        batch_size=128,
                        verbose=1,
                        use_multiprocessing=True,
                        workers=4,
                        # validation_split=0.3,
                        validation_data=(test_images, test_labels),
                        callbacks=[tensorboard_callback,
                                   early_stopping])
    model.save("./result/%s.h5" % m_name)
    hs = pd.DataFrame(history.history)
    hs["model"] = m_name
    result_loss.append(hs)
    y_true = le.inverse_transform(test_labels)
    y_predict = le.inverse_transform(np.argmax(model.predict(test_images), axis=1))
    result_predict.append(pd.DataFrame({"y_true": y_true, "y_predict": y_predict, "model": m_name}))

result_loss = pd.concat(result_loss, ignore_index=True, sort=False)
result_predict = pd.concat(result_predict, ignore_index=True, sort=False)
result_loss.to_csv("./result/result_loss.csv", index=False)
result_predict.to_csv("./result/result_predict.csv", index=False)

result_predict = pd.read_csv("./result/result_predict.csv")
result_loss = pd.read_csv("./result/result_loss.csv")
plt.figure()
for g_name, g_data in result_loss.groupby("model"):
    plt.plot(np.arange(len(g_data)) + 1, g_data['val_accuracy'].tolist(), label=g_name)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend()
plt.savefig("./result/测试集正确率分模型迭代.png")
plt.close()

import matplotlib
for g_name, g_data in result_predict.groupby("model"):
    print(g_name)
    cm = pd.pivot_table(g_data, index="y_predict", columns="y_true", values="model", aggfunc="count", fill_value=0)
    fig = plt.figure(figsize=[16, 12])
    axes = fig.add_subplot(1,1,1)
    plt.grid(False)
    image, cbar = heatmap(cm, cm.columns,  cm.columns, ax=axes,
                          cmap="Blues")
    texts = annotate_heatmap(image, valfmt="{x:}")
    plt.xlabel("真实标签")
    plt.ylabel("预测标签")
    plt.savefig("./result/graph/%s混淆矩阵.png" % g_name, dpi=300)
    plt.close()
    print(accuracy_score(g_data["y_true"], g_data["y_predict"]))
    print(classification_report_imbalanced(g_data["y_true"], g_data["y_predict"]))
