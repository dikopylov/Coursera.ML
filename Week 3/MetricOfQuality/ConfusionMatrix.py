import pandas
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve

data = pandas.read_csv('classification.csv')
true = data['true']
pred = data['pred']

# def conf_matr(k):
#     TP = FP = FN = TN = 0
#     for i in range(0, k):
#         if true[i] == pred[i] == 1:
#             TP += 1
#         elif true[i] == 0 and pred[i] == 1:
#             FP += 1
#         elif true[i] == 1 and pred[i] == 0:
#             FN += 1
#         elif true[i] == pred[i] == 0:
#             TN += 1
#     return TP, FP, FN, TN
#
# file = open("conf_matr.txt", "w")
# file.write(str(conf_matr(pred.size)))
# file.close()
#
# file = open("score.txt", "w")
# file.write(str(round(accuracy_score(pred, true),2 )) + str(round(precision_score(pred, true),2 )) +
#            str(round(recall_score(pred, true),2 )) + str(round(f1_score(pred, true),2 )))
# file.close()


file = open("max_roc.txt", "w")

data = pandas.read_csv('scores.csv')

score_logreg = data['score_logreg']
score_svm = data['score_svm']
score_knn = data['score_knn']
score_tree= data['score_tree']

scores = {}
for clf in data.columns[1:]:
    scores[clf] = roc_auc_score(data['true'], data[clf])

max_score = pandas.Series(scores).sort_values(ascending=False).head(1).index[0]

file.write(max_score)
file.close()

prc_logreg = precision_recall_curve(data['true'], score_logreg)
prc_svm = precision_recall_curve(data['true'], score_svm)
prc_knn = precision_recall_curve(data['true'], score_knn)
prc_tree = precision_recall_curve(data['true'], score_tree)

prc = [prc_logreg, prc_svm, prc_knn, prc_tree]

max_arr = []
for pr in prc:
    max = 0

    for i in range(0, pr[1].size):
        if pr[1][i] > 0.7:
            if max < pr[0][i]:
                max = pr[0][i]

    max_arr.append(max)

print(max_arr)


