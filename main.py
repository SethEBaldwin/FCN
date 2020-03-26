from fcn import FCN

# set PATH to location of Pacal VOC dataset
PATH = '/home/seth/Datasets/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/'

fcn = FCN(path=PATH)
#fcn.load('my_model')
fcn.train(epochs=75)
fcn.save('my_model')
fcn.evaluate(val=False)
fcn.evaluate()
for id in fcn.train_list[:10]:
    fcn.data.show_seg(id)
    fcn.predict(id)
for id in fcn.val_list[:10]:
    fcn.data.show_seg(id)
    fcn.predict(id)
