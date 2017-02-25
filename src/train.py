import tensorflow as tf
import numpy as np
from scipy import ndimage
from math import cos, sin, pi

n_classes=2
test_label_file='/home/vaibhavs/Projects/LungCancerDetection/data/stage1_sample_submission.csv'



#Load all avilable training data
input_data=np.load('imgs_train_data.npy').astype(np.float32)
input_labels=np.load('train_labels.npy').astype(np.float32)

# Determine the shape of the images
t, ZDIM, XDIM, YDIM = input_data.shape
# Downsampling fractions
ZDS, XDS, YDS = [2, 2, 2]

# Number of input channels
input_channels=1

# Create training/test/validation splits
train_data=input_data[:-300]
#.reshape((-1,ZDIM,XDIM,YDIM,input_channels))
train_labels=input_labels[:-300]

val_data=input_data[-300:-100].reshape((-1,ZDIM,XDIM,YDIM,input_channels))
val_labels=input_labels[-300:-100]

test_data=input_data[-100:].reshape((-1,ZDIM,XDIM,YDIM,input_channels))
test_labels=input_labels[-100:]


sub_data=np.load('imgs_test_data.npy').astype(np.float32).reshape(-1,ZDIM,XDIM,YDIM,input_channels)
sub_labels=np.ndarray((sub_data.shape[0], n_classes))
# Training parameters
keep_rate=0.7
batch_size=20

sub_ids=np.load('test_ids.npy')
sub_pred_dict={}
c_in=0.5*np.array(train_data.shape[2:4])
c_out=c_in

def training_data_aug(train_data, train_labels):
    t, ZDIM, XDIM, YDIM = train_data.shape
    x = train_data
    y = train_labels
    input_channels=1
    step=10 #degree radians
    for r in xrange(360/step):
        print ("Rotation Iteration %s ...." %r)
        a=r*step*pi/180
        transform=np.array([[cos(a),-sin(a)],[sin(a),cos(a)]])
        offset=c_in-c_out.dot(transform)
        for p in xrange(train_data.shape[0]):
            imageSet=train_data[p,:,:,:]
            for h in xrange(train_data.shape[1]):
                dst=ndimage.interpolation.affine_transform(train_data[p,h,:,:],
                        transform.T,order=2,offset=offset,output_shape=train_data.shape[2:4],cval=0.0,output=np.float32)
                imageSet=np.concatenate((imageSet,dst.reshape((-1,train_data.shape[2],
                    train_data.shape[3]))), axis=0)
            x=np.concatenate((x,imageSet.reshape((-1,
                train_data.shape[1],train_data.shape[2],train_data.shape[3]))), axis=0)
            y=np.concatenate((y,train_labels[p,:].reshape((-1,train_labels.shape[1]))), axis=0)

    return x.reshape((-1,ZDIM,XDIM,YDIM,input_channels)) ,y







def conv3d(x,W):
    return tf.nn.conv3d(x, W, strides=[1,1,1,1,1], padding='SAME')

def maxpool3d(x):
    return tf.nn.max_pool3d(x, ksize=[1,ZDS,XDS,YDS,1], strides=[1,ZDS,XDS,YDS,1],
            padding='SAME')


graph = tf.Graph()
with graph.as_default():
    # TODO: Try None for the batch_size
    tf_data = tf.placeholder(tf.float32, shape=[None, ZDIM, XDIM, YDIM, 1])
    tf_labels  = tf.placeholder(tf.float32, shape=[None, n_classes])


# Network Specification
#
#
    def model(data):

        DEPTH_1=32
        DEPTH_2=32
        ZDIM_1, XDIM_1, YDIM_1 = [ZDIM/ZDS, XDIM/XDS,
                YDIM/YDS]
        ZDIM_2, XDIM_2, YDIM_2 = [ZDIM_1/ZDS, XDIM_1/XDS,
                YDIM_1/YDS]
        # Why does fully connected dimension does not include the both the depth parameters is not clear ???
        FCDIM_I = ZDIM_2*XDIM_2*YDIM_2*DEPTH_1
        FCDIM_O = 1024


        weights = { 'W_conv1' : tf.Variable(tf.random_normal([3,3,3,1,DEPTH_1])),
                    'W_conv2' : tf.Variable(tf.random_normal([3,3,3,DEPTH_1,DEPTH_2])),
                    'W_fc'    : tf.Variable(tf.random_normal([FCDIM_I,FCDIM_O])),
                    'out'     : tf.Variable(tf.random_normal([FCDIM_O,n_classes]))}

        biases = {  'b_conv1' : tf.Variable(tf.random_normal([DEPTH_1])),
                    'b_conv2' : tf.Variable(tf.random_normal([DEPTH_2])),
                    'b_fc'    : tf.Variable(tf.random_normal([FCDIM_O])),
                    'out'     : tf.Variable(tf.random_normal([n_classes]))}

        conv1 = tf.nn.relu(conv3d(data,weights['W_conv1']) + biases['b_conv1'])
        conv1 = maxpool3d(conv1)

        conv2 = tf.nn.relu(conv3d(conv1,weights['W_conv2']) + biases['b_conv2'])
        conv2 = maxpool3d(conv2)
        fc = tf.reshape(conv2, [-1,FCDIM_I])
        fc = tf.nn.relu(tf.matmul(fc,weights['W_fc']) + biases['b_fc'])
        fc = tf.nn.dropout(fc,keep_rate)
        logits = tf.matmul(fc, weights['out']) +  biases['out']

        return logits

    #loss, optimizer and prediction
    logits = model(tf_data)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits,tf_labels))
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)
    prediction = tf.nn.softmax(logits)


num_steps = 500
def accuracy(predictions, labels):
    return (100.0*np.sum(np.argmax(predictions,1) == np.argmax(labels,1))/predictions.shape[0])

#
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.5)
#with tf.Session(graph=graph,config = tf.ConfigProto(gpu_options = gpu_options)) as session:



# Training

best_acc=0
checkPointFile = './model.checkPoint'
with tf.Session(graph=graph) as session:
    saver = tf.train.Saver()
    tf.global_variables_initializer().run()
    print('Initialized')
    train_data, train_labels=training_data_aug(train_data,train_labels)
    for step in range(num_steps):
        offset = (step*batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_data[offset:(offset+batch_size), :, :, :, :]
        batch_labels = train_labels[offset:(offset+batch_size), :]
        feed_dict = { tf_data: batch_data, tf_labels: batch_labels }
        _,l, p = session.run(
                [optimizer, loss, prediction], feed_dict=feed_dict
                )
        if (step % 10 == 0) :
            print('Minibatch loss at step %d: %f' %(step, l))
            print('Minibatch accuracy: %f' %accuracy(p,batch_labels))
            p = session.run(prediction, {tf_data: val_data, tf_labels: val_labels})

            val_acc=accuracy(p, val_labels)
            if val_acc > best_acc:
                best_acc = val_acc
                saver.save(session, checkPointFile)
            print('Validation accuracy: %f' %val_acc)
    test_predictions = session.run(prediction,
                    {tf_data: test_data, tf_labels: test_labels})
    print('Test Accuracy: %f' %accuracy(test_predictions, test_labels))

#loading the best model
    saver.restore(session, checkPointFile)

    # Prediction of submission images
    sub_predictions = session.run(prediction, {tf_data:sub_data,
        tf_labels:sub_labels})
    for i in range(sub_ids.shape[0]):
        sub_pred_dict[sub_ids[i]]=sub_predictions[i][1]

#    print sub_predictions
    g=open("submission.csv", 'w')
    g.write("id,cancer\n");
    with open(test_label_file, 'r') as f:
        for aline in f:
            tag, p = aline.rstrip().split(',')
            if tag == 'id' : continue
            g.write("%s,%s\n" %(tag,sub_pred_dict[tag]))
    g.close()


