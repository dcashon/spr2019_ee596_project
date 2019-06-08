from bddmodel import BDDModel, SimpleBDDModel
import tensorflow as tf
from bddfunc import training_batcher
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import datetime

tf.reset_default_graph()
my_model = SimpleBDDModel()


batch_size = 128
num_data = 13 # number of saved batch data
num_epochs = 5


# Train Model
init = tf.global_variables_initializer()
saver = tf.train.Saver()
enc = OneHotEncoder(sparse=False, categories=[np.arange(10)])
day_path = str(datetime.datetime.now())[:10]
counter = 0
with tf.Session() as sess:
    init.run()
    for i in range(num_epochs):
        print('Training on Epoch \t' + str(i))
        for j in range(13): # 13 saved data
            print('Training on Dataset \t' + str(j))
            batcher = training_batcher(j, batch_size)
            for data, labels in batcher:
                counter += 1
                temp = enc.fit_transform(np.reshape(labels, (-1,1)))
                f_dict = {my_model.X: data / 255, my_model.Y1: temp}
                my_model.minimizer.run(feed_dict=f_dict)
                if counter % 10 == 0:
                    print(my_model.loss.eval(feed_dict=f_dict))
                    print(my_model.accuracy.eval(feed_dict=f_dict))
            save_path = saver.save(sess, "./tmp/model_job_" + day_path + "_epoch" + str(i) + "_set" + str(j) + ".ckpt")
            print("Model saved in path: %s" % save_path)

