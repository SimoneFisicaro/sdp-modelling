import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, auc, roc_curve
from os import remove, path, makedirs


def display_curve(fit_results, title, curve_type):
    metrics = fit_results.history[curve_type]
    val_metrics = fit_results.history['val_' + curve_type]

    epochs = range(1, len(metrics) + 1)

    fig, ax = plt.subplots()
    plt.plot(epochs, metrics, 'bo', label='Training')
    plt.plot(epochs, val_metrics, 'b', label='Validation')
    plt.legend()
    plt.figure()
    plt.title(title)
    plt.ylabel(curve_type)
    plt.xlabel('Epoch')

    return fig


def diplay_confusion_matrix(y_test, y_pred, classes):
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)

    plt.figure(figsize=(5.5, 4))
    fig = sns.heatmap(cm_df, annot=True)
    plt.title(' \nAccuracy:{0:.3f}'.format(accuracy_score(y_test, y_pred)))
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    return fig


def display_roc(y_test, y_pred):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    auc_val = auc(fpr, tpr)

    fig, ax = plt.subplots()
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='Model (area = {:.3f})'.format(auc_val))
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()

    return fig


def smooth_curve(points, factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


def save_figure_to_mlflow(figure, figure_name, temp_folder):
    if not path.exists(temp_folder):
        makedirs(temp_folder)

    img_path = path.join(temp_folder, figure_name + '_img.png')
    figure.savefig(img_path)
    mlflow.log_artifact(img_path, "images")
    remove(img_path)
