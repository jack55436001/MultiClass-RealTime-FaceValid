"""FaceValid core code"""

# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import tensorflow as tf
import numpy as np
import sys
import os
import argparse
import facenet
import align.detect_face
import facial_landmarks as fl
import cv2
from os import listdir
from os.path import isfile, join
import imghdr

def inference(args):
    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor	
    mypath = args.image_path

    print('Creating Mtcnn networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

    print('Creating Facenet networks and loading parameters')
    with tf.Graph().as_default():

        with tf.Session() as sess:
      
            # Load the model
            facenet.load_model(args.model)
    
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # Read image from database and inference
            onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
            myImage_dict = {}
            myImage = []
            index = 0
            for file in onlyfiles:
                isImage = None
                name = file
                file = mypath + '/' + file
                isImage = imghdr.what(file)
                if isImage != None:
                    myImage.append(file)
                    name = name.split('.')[0]
                    myImage_dict[index] = name
                    index+=1
            images = load_and_align_data(myImage, args.image_size, args.margin, pnet, rnet, onet)
            feed_dict = { images_placeholder: images, phase_train_placeholder:False }
            emb_database = sess.run(embeddings, feed_dict=feed_dict)

            #Start opencv Camera
            cap = cv2.VideoCapture(0)
			
            while(True):
                # Capture frame-by-frame
                ret, frame = cap.read()
                bounding_boxes, _ = align.detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
                checkbbox = np.copy(bounding_boxes)
                
                # for idx in range (checkbbox.shape[0]):
                # 	bbox = checkbbox[idx].astype(int)
                # 	if bbox[3] - bbox[1] <= 0 or bbox[2] - bbox[0] <= 0 or :
                # 		print('Delete Invalid bbox')
                # 		np.delete(bounding_boxes,idx,0)

                if bounding_boxes.shape[0] > 0:
                    #finding face
                    capture = align_single_data(frame, args.image_size, args.margin, pnet, rnet, onet , bounding_boxes)
                    feed_dict = { images_placeholder: capture, phase_train_placeholder:False }
                    emb_capture = sess.run(embeddings, feed_dict=feed_dict)			    	
                    for i in range(emb_capture.shape[0]):
                        closest = 10
                        valid = -1;
                        for j in range(emb_database.shape[0]):
                            dist = np.sum(np.square(np.subtract(emb_capture[i,:], emb_database[j,:])))
                            if dist < closest:
                                closest = dist
                                valid = j
                        cv2.rectangle(frame, (int(bounding_boxes[i][0]), 
                        int(bounding_boxes[i][1])), 
                        (int(bounding_boxes[i][2]), int(bounding_boxes[i][3])), 
                        (0, 255, 0), 2)
                        if closest < 0.8:
                            cv2.putText(frame,myImage_dict[valid], (int(bounding_boxes[i][0]),int(bounding_boxes[i][3])), 
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255, 0 ,0), 
                            thickness = 2, lineType = 2)
                        else :
                            cv2.putText(frame,'Warning Not in Database', (int(bounding_boxes[i][0]),int(bounding_boxes[i][3])), 
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255, 0 ,0), 
                            thickness = 2, lineType = 2)                        	
			    	
			    # Display the resulting frame
                cv2.imshow('frame',frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

			# When everything done, release the capture
            cap.release()
            cv2.destroyAllWindows()

            
            # nrof_images = len(imageDir)

            # print('Images:')
            # for i in range(nrof_images):
            #     print('%1d: %s' % (i, imageDir[i]))
            # print('')
            
            # # cal  distance matrix
            # print('Distance  matrix')
            # print('    ', end='')
            # for i in range(nrof_images):
            #     print('    %1d     ' % i, end='')
            # print('')
            # for i in range(nrof_images):
            #     print('%1d  ' % i, end='')
            #     for j in range(nrof_images):
            #         dist = np.sqrt(np.sum(np.square(np.subtract(emb[i,:], emb[j,:]))))
            #         print('  %1.4f  ' % dist, end='')
            #     print('')
            
            # # cal square distance matrix
            # print('Distance square matrix')
            # print('    ', end='')
            # for i in range(nrof_images):
            #     print('    %1d     ' % i, end='')
            # print('')
            # for i in range(nrof_images):
            #     print('%1d  ' % i, end='')
            #     for j in range(nrof_images):
            #         dist = np.sum(np.square(np.subtract(emb[i,:], emb[j,:])))
            #         print('  %1.4f  ' % dist, end='')
            #     print('')

            # # #cal cosine simularity
            # print('Consine Simularity matrix')
            # print('    ', end='')
            # for i in range(nrof_images):
            #     print('    %1d     ' % i, end='')
            # print('')
            # for i in range(nrof_images):
            #     print('%1d  ' % i, end='')
            #     for j in range(nrof_images):
            #         #dist = np.sqrt(np.sum(np.square(np.subtract(emb[i,:], emb[j,:]))))
            #         dist = np.sum(np.dot(emb[i,:], emb[j,:])) / (np.sqrt(np.sum(np.square(emb[i,:]))) * np.sqrt(np.sum(np.square(emb[j,:]))))
            #         print('  %1.4f  ' % dist, end='')
            #     print('')

            #cal correlation coefficient
            # mean_List = []
            # print('Correlation coefficient matrix')
            # print('    ', end='')
            # for i in range(nrof_images):
            #     print('    %1d     ' % i, end='')
            #     mean_List.append(np.mean(emb[i,:]))
            # print('')
            # for i in range(nrof_images):
            #     print('%1d  ' % i, end='')
            #     for j in range(nrof_images):
            #         #dist = np.sqrt(np.sum(np.square(np.subtract(emb[i,:], emb[j,:]))))
            #         nu = np.sum(np.dot(np.subtract(emb[i,:],mean_List[i]) , np.subtract(emb[j,:],mean_List[j])))
            #         de = np.sqrt(np.sum(np.square(np.subtract(emb[i,:],mean_List[i])))) * np.sqrt(np.sum(np.square(np.subtract(emb[j,:],mean_List[j]))))
            #         dist = nu / de
            #         print('  %1.4f  ' % dist, end='')
            #     print('')
def align_single_data(image,image_size, margin,  pnet, rnet, onet , bounding_boxes):
	output = [None] * bounding_boxes.shape[0]

	for numFace in range(bounding_boxes.shape[0]):
	    #img = misc.imread(os.path.expanduser(image))   
	    img_size = np.asarray(image.shape)[0:2]
	    det = np.squeeze(bounding_boxes[numFace,0:4])
	    bb = np.zeros(4, dtype=np.int32)
	    bb[0] = np.maximum(det[0]-margin/2, 0)
	    bb[1] = np.maximum(det[1]-margin/2, 0)
	    bb[2] = np.minimum(det[2]+margin/2, img_size[1])
	    bb[3] = np.minimum(det[3]+margin/2, img_size[0])
        #fix bbox error
        if bb[1] > bb[3]:
            bb[1] , bb[3] =  bb[3] , bb[1]
        if bb[0] > bb[2]:
            bb[0] , bb[2] =  bb[2] , bb[0]   
	    cropped = image[bb[1]:bb[3],bb[0]:bb[2],:]
	    aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
	    prewhitened = facenet.prewhiten(aligned)
	    output[numFace] = prewhitened
	    

	return output   	

def load_and_align_data(image_paths, image_size, margin,  pnet, rnet, onet):

    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor
    
    # print('Creating networks and loading parameters')
    # with tf.Graph().as_default():
    #     gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
    #     sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    #     with sess.as_default():
    #         pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
  
    nrof_samples = len(image_paths)
    img_list = [None] * nrof_samples
    for i in range(nrof_samples):
        img = misc.imread(os.path.expanduser(image_paths[i]))   
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        det = np.squeeze(bounding_boxes[0,0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0]-margin/2, 0)
        bb[1] = np.maximum(det[1]-margin/2, 0)
        bb[2] = np.minimum(det[2]+margin/2, img_size[1])
        bb[3] = np.minimum(det[3]+margin/2, img_size[0])
        #fix bbox error
        if bb[1] > bb[3]:
            bb[1] , bb[3] =  bb[3] , bb[1]
        if bb[0] > bb[2]:
            bb[0] , bb[2] =  bb[2] , bb[0]   
        cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
        aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        #aligned = fl.faceLandMarks(aligned)
        prewhitened = facenet.prewhiten(aligned)
        img_list[i] = prewhitened

        #cv2.imwrite(str(i)+'.jpg',prewhitened)
        #misc.imsave(str(i)+'.jpg',img_list[i])
        # cv2.imshow('preWhiten',img_list[i])
        # cv2.waitKey(0)
    images = np.stack(img_list)

    return images

