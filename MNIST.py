import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib


def export_model_for_mobile(model_name, input_node_name, output_node_name, sess):
    # dump graph as pbtxt
    tf.train.write_graph(sess.graph_def, 'out', model_name + '_graph.pbtxt')
    # save checkpoint
    tf.train.Saver().save(sess, 'out/' + model_name + '.chkp')
    # freeze the graph
    # param1: path for pbtxt, param2: path for checkpoint, param3: name for last layer
    freeze_graph.freeze_graph('out/' + model_name + '_graph.pbtxt', None,
        False, 'out/' + model_name + '.chkp', output_node_name,
        "save/restore_all", "save/Const:0",
        "out" + "/frozen_" + model_name + '.pb', True, "")
    # parse graph to optimize
    input_graph_def = tf.GraphDef()
    with tf.gfile.Open("out"+ "/frozen_" + model_name + '.pb', "rb") as f:
        input_graph_def.ParseFromString(f.read())
    # optimize
    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
            input_graph_def, [input_node_name], [output_node_name],
            tf.float32.as_datatype_enum)
    # output lite model
    with tf.gfile.FastGFile("out"+ '/tensorflow_lite_' + model_name + '.pb', "wb") as f:
        f.write(output_graph_def.SerializeToString())


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# X is our placeholder value for our input
x = tf.placeholder(tf.float32, [None, 784], name="x")
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b, name="y")

y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Create an InteractiveSession
sess = tf.InteractiveSession()

# Create an operation to initialize the variables we created
tf.global_variables_initializer().run()
for i in range(100001):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  if i % 1000 == 0:
      print("i = ",i,sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
export_model_for_mobile("model", "x", "y", sess)