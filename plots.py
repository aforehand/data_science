import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.tree import export_graphviz
from PIL import Image
import subprocess
from sklearn.metrics import roc_curve, auc
from matplotlib import cm
import seaborn as sns
import warnings


def class_regions(X, y, model, title='', subplot=plt, ticks=5, round=False, point_size=20):
    h = 0.02 # step size in the mesh

    # Plot the decision boundary. For that, we will asign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    x_min, x_max = X[:,0].min() - .5, X[:,0].max() + .5
    y_min, y_max = X[:,1].min() - .5, X[:,1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)

    ax = subplot
    if subplot == plt:
        ax = plt.gca()

    rainbow = [list(plt.cm.rainbow(i))[:3] for i in range(256)]
    colors = rainbow[::int((len(rainbow)/len(model.classes_)))]
    my_cmap = LinearSegmentedColormap.from_list('class_map', colors)

    #if targets are strings, map to ints for color plot
    try:
        cbar = ax.pcolormesh(xx, yy, Z, cmap=my_cmap)
    except AttributeError as e:
        mapping = dict(zip(np.unique(Z), np.arange(len(np.unique(Z)))))
        Z = np.vectorize(mapping.get)(Z)
        cbar = ax.pcolormesh(xx, yy, Z, cmap=my_cmap)

    warnings.simplefilter('ignore', UserWarning)
    # plot the training points
    x_step_size = (x_max - x_min) / ticks
    y_step_size = (y_max - y_min) / ticks
    for i in range(len(model.classes_)):
        idx = np.where(y==model.classes_[i])
        ax.scatter(X[idx,0], X[idx,1], c=colors[i], s=point_size, edgecolor='k', label=model.classes_[i])

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(np.arange(x_min, x_max, x_step_size))
    ax.set_yticks(np.arange(y_min, y_max, y_step_size))

    if subplot==plt:
        plt.suptitle(title)
    else:
        ax.set_title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    if subplot == plt:
        plt.show()


def validation_curve(train_scores, test_scores, model_name='', param=''):
    plt.figure()

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.title('Validation Curve with '+model_name)
    plt.xlabel(param)
    plt.ylabel('Score')
    plt.ylim(0.0, 1.1)
    lw = 2

    plt.semilogx(param_range, train_scores_mean, label='Training score',
                 color='darkorange', lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color='darkorange', lw=lw)
    plt.semilogx(param_range, test_scores_mean, label='Cross-validation score',
                 color='navy', lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color='navy', lw=lw)
    plt.legend(loc='best')
    plt.show()


def decision_tree(model, feature_names=None, target_names=None, file_name='tree'):
    export_graphviz(model, out_file=file_name+'.dot', feature_names=feature_names, class_names=target_names, filled=True, impurity=False, rounded=True)

    command = ['dot', '-Tpng', file_name+'.dot', '-o', file_name+'.png']
    subprocess.check_call(command)

    Image.open(file_name+'.png').show()

def feature_importances(importances, feature_names, forest=False, truncate=True):
    # importances = model.feature_importances_
    importances, feature_names = zip(*sorted(zip(importances, feature_names)))

    if(forest):
        std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    else:
        std = 0

    if(truncate):
        importances = [x for x in importances if x != 0]
        feature_names = feature_names[-len(importances):]

    plt.figure()
    plt.title('Feature Importances')
    patches = plt.barh(np.arange(len(feature_names)), importances, xerr=std, color='red', align='center')
    plt.yticks(np.arange(len(feature_names)), feature_names)
    plt.xlabel('Importance')
    plt.ylabel('Feature')

    cm = plt.cm.get_cmap('jet')
    # idfk how this used to work
    # col = importances - min(importances)
    # col /= max(col)
    max_importance = max(importances)
    col = [x/max_importance for x in importances]

    for c,p in zip(col, patches):
        plt.setp(p, 'facecolor', cm(c))

    plt.show()

def precision_recall(precision, recall, thresholds):
    closest_zero = np.argmin(np.abs(thresholds))
    closest_zero_p = precision[closest_zero]
    closest_zero_r = recall[closest_zero]

    plt.figure()
    plt.xlim([0.0, 1.01])
    plt.ylim([0.0, 1.01])
    plt.plot(precision, recall, label='Precision-Recall Curve')
    plt.plot(closest_zero_p, closest_zero_r, 'o', markersize = 12, fillstyle = 'none', c='r', mew=3)
    plt.xlabel('Precision', fontsize=16)
    plt.ylabel('Recall', fontsize=16)
    plt.axes().set_aspect('equal')
    plt.show()

def roc(y_true, y_predict, title=''):
    fpr, tpr, _ = roc_curve(y_true, y_predict)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.xlim([-0.01, 1.00])
    plt.ylim([-0.01, 1.01])
    plt.plot(fpr, tpr, lw=3, label='ROC curve (area = {:0.2f})'.format(roc_auc))
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('ROC curve '+title, fontsize=16)
    plt.legend(loc='lower right', fontsize=13)
    plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
    plt.axes().set_aspect('equal')
    plt.show()

def roc_gamma_sweep(X, y, gamma_list, title=''):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    plt.figure()
    plt.xlim([-0.01, 1.00])
    plt.ylim([-0.01, 1.01])
    for g in gamma_list:
        svm = SVC(gamma=g).fit(X_train, y_train)
        y_score_svm = svm.decision_function(X_test)
        fpr_svm, tpr_svm, _ = roc_curve(y_test, y_score_svm)
        roc_auc_svm = auc(fpr_svm, tpr_svm)
        accuracy_svm = svm.score(X_test, y_test)
        print("gamma = {:.2f}  accuracy = {:.2f}   AUC = {:.2f}".format(g, accuracy_svm, roc_auc_svm))
        plt.plot(fpr_svm, tpr_svm, lw=3, alpha=0.7,
                 label='SVM (gamma = {:0.2f}, area = {:0.2f})'.format(g, roc_auc_svm))

    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate (Recall)', fontsize=16)
    plt.plot([0, 1], [0, 1], color='k', lw=0.5, linestyle='--')
    plt.legend(loc="lower right", fontsize=11)
    plt.title('ROC curve ' +title, fontsize=16)
    plt.axes().set_aspect('equal')

    plt.show()

def confusion_matrix(y_true, y_predict, target_names, model_name=''):
    confusion_mc = confusion_matrix(y_true, y_predict)
    df_cm = pd.DataFrame(confusion_mc, index = [i for i in range(0,len(target_names))], columns = [i for i in range(0,len(target_names))])
    plt.figure(figsize=(5.5,4))
    sns.heatmap(df_cm, annot=True)
    plt.title(model_name+'  \nAccuracy:{0:.3f}'.format(accuracy_score(y_true, y_predict)))
    plt.yticks(np.arange(len(target_names)), target_names)
    plt.xticks(np.arange(len(target_names)), target_names)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
