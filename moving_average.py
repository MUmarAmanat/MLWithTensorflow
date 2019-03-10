import tensorflow as tf 
import numpy as np 

# dummy data
raw_data = np.random.normal(10, 1, 100)

# defining graph
# define vaiable, placeholder and constant for graph
alpha = tf.constant(0.05)
prev_average = tf.Variable(initial_value=0, dtype=tf.float32, name="prev_average")
curr_value = tf.placeholder(dtype=tf.float32, name="curr_value")

# initialize op for intiializing all variable in graph
init = tf.global_variables_initializer()

# now defining operations
update_avg = (1 - alpha) * prev_average + curr_value * alpha

# create some summary to visualize in tensorboard
# average history
avg_hist = tf.summary.scalar("running_average", update_avg)
curr_hist = tf.summary.scalar("current_value", curr_value)
# now merge all above summaries so we cann call only s1ngle operation during sess.run
merged = tf.summary.merge_all()
# make writer object which will write summary and graph for visualization prpose
writer = tf.summary.FileWriter("./log")

with tf.Session() as sess:
    sess.run(init)
    writer.add_graph(sess.graph)
    for i in range(len(raw_data)):
        summaries, curr_avg = sess.run([merged, update_avg], feed_dict={curr_value: raw_data[i]})
        sess.run(tf.assign(prev_average, curr_avg))
        writer.add_summary(summaries, i)
