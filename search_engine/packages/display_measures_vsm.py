import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable

a=np.load('model/precision_by_percent_recall.npy')

precision = a[1]
recall = a[0]
AP = a[2]
mAP = a[3]

table = PrettyTable(float_format="6.4")
table.add_column('Recall', np.array(recall).round(4))
table.add_column('Precision', np.array(precision).round(4))
print(table)
print('Average Precision (AP): ' + str(np.round(AP, 4)))
print('Mean Average Precision (MAP): ' + str(np.round(mAP, 4)))

plt.plot(recall, precision)
plt.ylabel('Precision')
plt.xlabel('Recall')

plt.show()

#f1 = [2 * precision[i] * recall[i] / (precision[i] + recall[i]) for i in range(len(precision))]


