from source.data_manager import extract_train_sets, extract_test_sets, rescale_all_sets, encode_all_train_sets, \
    encode_all_test_sets, encode_sgrna_sequence
from source.cnn import cnn_ontar_reg_model, prepare_set
from sklearn.model_selection import KFold, train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import numpy as np

extract_train_sets()
extract_test_sets()
rescale_all_sets()

early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=50)

model_293t = cnn_ontar_reg_model()
model_mel4 = cnn_ontar_reg_model()
model_hl60 = cnn_ontar_reg_model()

train_sets_array = encode_all_train_sets()
test_sets_array = encode_all_test_sets()

features_293t, labels_293t = prepare_set(train_sets_array[0])
features_mel4, labels_mel4 = prepare_set(train_sets_array[1])
features_hl60, labels_hl60 = prepare_set(train_sets_array[2])

x_train, x_test, y_train, y_test = train_test_split(features_hl60, labels_hl60, test_size=0.1)
his = model_hl60.fit(x=x_train, y=y_train, epochs=250, verbose=0, validation_data=(x_test, y_test))
model_hl60.evaluate(x=x_test, y=y_test)

y_pred = model_hl60.predict(x=features_hl60)
score, _ = spearmanr(labels_hl60, y_pred)
print("Spearman:", score)

y_pred = model_hl60.predict(x=features_293t)
score, _ = spearmanr(labels_293t, y_pred)
print("Spearman:", score)

y_pred = model_hl60.predict(x=features_mel4)
score, _ = spearmanr(labels_mel4, y_pred)
print("Spearman:", score)


plt.plot(his.history['loss'])
plt.plot(his.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

x_train, x_test, y_train, y_test = train_test_split(features_mel4, labels_mel4, test_size=0.1)
his = model_mel4.fit(x=x_train, y=y_train, epochs=250, verbose=0, validation_data=(x_test, y_test))
model_mel4.evaluate(x=x_test, y=y_test)

y_pred = model_mel4.predict(x=features_hl60)
score, _ = spearmanr(labels_hl60, y_pred)
print("Spearman:", score)

y_pred = model_mel4.predict(x=features_293t)
score, _ = spearmanr(labels_293t, y_pred)
print("Spearman:", score)

y_pred = model_mel4.predict(x=features_mel4)
score, _ = spearmanr(labels_mel4, y_pred)
print("Spearman:", score)

plt.plot(his.history['loss'])
plt.plot(his.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

x_train, x_test, y_train, y_test = train_test_split(features_293t, labels_293t, test_size=0.1)
his = model_293t.fit(x=x_train, y=y_train, epochs=250, verbose=0, validation_data=(x_test, y_test))
model_293t.evaluate(x=x_test, y=y_test)

y_pred = model_293t.predict(x=features_hl60)
score, _ = spearmanr(labels_hl60, y_pred)
print("Spearman:", score)

y_pred = model_293t.predict(x=features_293t)
score, _ = spearmanr(labels_293t, y_pred)
print("Spearman:", score)

y_pred = model_293t.predict(x=features_mel4)
score, _ = spearmanr(labels_mel4, y_pred)
print("Spearman:", score)

plt.plot(his.history['loss'])
plt.plot(his.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

"""
train_sets_array = encode_all_train_sets()

for elem in train_sets_array:
    cv = KFold(n_splits=10, shuffle=True)
    val_loss = []
    history_loss = []
    spearman = 0
    ls = 0
    features, labels = prepare_set(elem)
    for train_index, val_index in cv.split(features, labels):
        model = cnn_ontar_reg_model()
        train_features, val_features, train_labels, val_labels = features[train_index], features[val_index], \
                                                                 labels[train_index], labels[val_index]
        results = model.fit(x=train_features, y=train_labels, epochs=250, verbose=0,
                            validation_data=(val_features, val_labels))
        l = model.evaluate(x=val_features, y=val_labels, verbose=1)

        y_pred = model.predict(x=val_features)

        score, _ = spearmanr(val_labels, y_pred)

        history_loss.append(results.history['loss'])
        val_loss.append(results.history['val_loss'])

        ls += l
        spearman += score

    loss = [0] * 250
    vloss = [0] * 250

    for i in range(250):
        for j in range(10):
            loss[i] += history_loss[j][i]
            vloss[i] += val_loss[j][i]
            j += 1
        loss[i] /= 10
        vloss[i] /= 10
        i += 1

    plt.plot(loss)
    plt.plot(vloss)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    print("Spearman:", spearman / 10, "Loss:", ls / 10)
"""
