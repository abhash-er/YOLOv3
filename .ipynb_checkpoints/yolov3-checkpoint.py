import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import BatchNormalization, Conv2D, Input, ZeroPadding2D, LeakyReLU, UpSampling2D

#print("Success")
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def parse_cfg(cfgfile):
    with open(cfgfile,'r') as file:
        lines = [line.rstrip('\n') for line in file if (line != '\n' and line[0] != '#') ]
        
    holder = {}
    blocks = []

    for line in lines:

        if line[0] == '[':

            line = 'type=' + line[1:-1].rstrip()

            if len(holder) != 0:
                blocks.append(holder)
                holder = {}

        key, value = line.split("=")
        holder[key.rstrip()] = value.lstrip()

    blocks.append(holder)
    return blocks
    
    
def YOLOv3Net(cfgfile,model_size,num_classes):

    blocks = parse_cfg(cfgfile)

    #cnfg file contains 5 type of layers , Convolutional , upsample, route , shortcut, yolo
    outputs = {}
    output_filters = []
    filters = []
    out_pred = []
    scale = 0

    inputs = input_image = Input(shape=model_size)
    inputs = inputs/255.

    for i,block in enumerate(blocks[1:]):

        #Convolutional--
        #there are 2 conv layer, one with batch normalization and other without batc normalization 
        #conv->BN->Leaky_relu
        #conv->Linear (last layer)

        if(block["type"] == "convolutional"):

            activation = block["activation"]
            filters = int(block["filters"])
            kernel_size = int(block["size"])
            strides = int(block["stride"])

            if strides > 1:
                inputs = ZeroPadding2D(((1,0),(1,0)))(inputs)

            inputs = Conv2D(filters,kernel_size,
                            strides=strides,
                            padding = 'valid' if strides > 1 else 'same',
                           name = 'conv_' + str(i),
                           use_bias = False if ("batch_normalize" in block) else True)(inputs)

            if "batch_normalize" in block:
                inputs = BatchNormalization(name ='bnorm_' + str(i))(inputs)
                inputs = LeakyReLU(0.1, name ='leaky_' + str(i) )(inputs)



        #Upsample layer
        #performs upsampling by a factor of stride  (bilinear upsampling method)

        elif(block["type"] == "upsample"):
            stride = int(block["stride"])
            inputs = UpSampling2D(stride)(inputs)

        #Route Layer
        #route has 2 type of attributes, 
        #single value (eg. -4)-> -x means to go backwards x layers and then output feature map
        #double values (eg. -1, 61)-> -x,y concatenate feature from x layer backward and y

        elif(block["type"] == "route"):
            block["layers"] = block["layers"].split(',')
            start = int(block["layers"][0])

            if len(block["layers"]) > 1:
                end = int(block["layers"][1])-i
                filters = output_filters[i+start] + output_filters[end]
                inputs = tf.concat([outputs[i+start],outputs[i+end]],axis=-1)

            else:
                filters = output_filters[i+start]
                inputs = outputs[i+start]

        #Shortcut layer (residual)

        elif(block["type"] == "shortcut"):

            from_ = int(block["from"])
            inputs = outputs[i-1] + outputs[i+from_]


        #YOLO Layer

        elif(block["type"] == "yolo"):

            mask = block["mask"].split(",")
            mask = [int(o) for o in mask]

            anchors = block["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            #make tuple
            anchors = [(anchors[i],anchors[i+1]) for i in range(0,len(anchors),2)]
            #qpply mask
            anchors = [anchors[i] for i in mask]
            n_anchors = len(anchors)

            #reshape in form of [None,B*grid_size*grid_size,5+C] 
            #B-> number of anchors
            #C->number of classes

            out_shape = inputs.get_shape().as_list()

            inputs = tf.reshape(inputs,[-1,n_anchors*out_shape[1]*out_shape[2],5+num_classes])

            #store centres(bx,by), shape (bh,bw),confidence_score, classes_probability  
            box_centres = inputs[:,:,0:2]
            box_shapes  = inputs[:,:,2:4]
            confidence  = inputs[:,:,4:5]
            classes     = inputs[:,:,5:num_classes+5]

            #convert in range 0-1

            box_centres = tf.sigmoid(box_centres)
            confidence = tf.sigmoid(confidence)
            classes = tf.sigmoid(classes)

            anchors = tf.tile(anchors,[out_shape[1] * out_shape[2],1])
            box_shapes = tf.exp(box_shapes) * tf.cast(anchors,dtype = tf.float32)

            x = tf.range(out_shape[1],dtype = tf.float32)
            y = tf.range(out_shape[2],dtype = tf.float32)

            cx,cy = tf.meshgrid(x,y)
            cx = tf.reshape(cx,(-1,1))
            cy = tf.reshape(cy,(-1,1))

            cxy = tf.concat([cx,cy],axis = -1)
            cxy = tf.tile(cxy,[1,n_anchors])
            cxy = tf.reshape(cxy,[1,-1,2])

            strides = (input_image.shape[1] // out_shape[1], input_image.shape[2] // out_shape[2])
            box_centres = (box_centres + cxy) * strides


            prediction = tf.concat([box_centres,box_shapes,confidence,classes],axis = -1)
            if scale:
                out_pred = tf.concat([out_pred,prediction],axis = 1)
            else:
                out_pred = prediction
                scale = 1

        outputs[i] = inputs
        output_filters.append(filters)


    model = Model(input_image,out_pred)
    model.summary()

    return model
                
                
                
                