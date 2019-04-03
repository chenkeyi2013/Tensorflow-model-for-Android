# Tensorflow-model-for-Android
this is a repository that shows how to retrain a tuned model with your own picture in Tensorflow and use it for a Android App.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------

Consider our situation:

	What we want to do?
		train a model that can solve our classify problem,and use it on Android.
	What we have?
		A few data photograph,maybe youself or your friend.
	Build your own nerual network or choose a tuned model and retrain it?
		A tuned model is a better choice I think.
		1.I don't read enough paper about the conv nerual network when I build these code,have no idea about what structures or parameters are useful intuitively.
		2.There are no enough data to feed.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------

The InceptionV3 model:

	This is a very commom example,and the code are build by the high framework of Tensorflow,so I dont want to explain it.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------

The mobilenet model:

	This maybe a solution for the pb file cost too much time on Andriod after I get my trained InceptionV3 model.

	The Tensorflow Lite website shows that a InceptionV3 quantized Lite file will cost 600ms in Andriod,and Top1 accuracy is about 79%,
	but a mobilenet pb file cost 100ms in Andriod and Top1 accuracy is about 71%.
	
	--------------------------------------------------------------------------------------------------------------------------------------------
	Data----->load to mermory----->Load the model structure----->Find a node with tensorboard----->build a simple FC layer by you own----->train


	Data:
		prepare your own pictures and change it to Tfrecords with /Tensorflow/python/item/data_to_tfrecord.py.
		example:
			your three types picture:	/home/picture/0,/home/picture/1,/home/picture/2
			change the parameter in py:	/home/picture/
			

		About Tfrecod:
				be careful about the parser defined.

	


	Load to mermory:
		
		change the dir in /Tensorflow/python/mobilenet_retrain,like:/home/train.tfrecord.
		and import the /Tensorflow/python/data_op.py to load tfrecords to your mermory as list.

	
	Load the model structure:

		Use the /Tensorflow/python/items/import_pb.py to import structure of a pb or ckpt file.
		
	Find a node with tensorboard:

		use these code to build your tensorboard log.

		-------------------------------------------------------
		writer = tf.summary.FileWriter(TENSORBOARD_LOG, g1)
		writer.close()
		-------------------------------------------------------
		and command 'tensorboard --logdir=log to overview the graph with tensorboard.
		you should check the model paper and find the FC layer or choose a node you want to begin.

	build a simple FC layer by you own:
		
		I tryed a serveral simple structure and write result in /Tensorflow/log
		there is a overfitting problem because my data is not good enough,so try feed better data or change another node before 'MobilenetV1/Logits/Dropout_1b/Identity:0'.
		I will update the result as soon as I solve this problem.

	train:

		nothing import,try your own optimizer and hyperparameters.
		  
      
